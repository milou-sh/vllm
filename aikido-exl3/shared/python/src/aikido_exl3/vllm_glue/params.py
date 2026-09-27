"""Collector parameters.

One zero-sized parameter per checkpoint suffix, named exactly like the suffix, so vLLM's unmodified
weight routing (`<layer>.<suffix>` + shard id) finds it. Shapes depend on each tensor's bitrate and are
unknown at create_weights time, so nothing is allocated up front; tensors are kept per shard id and
turned into plain parameters in process_weights_after_loading.
"""
from __future__ import annotations

import torch

from .compat import BasevLLMParameter


def _collect(param: "Exl3Collector", loaded_weight: torch.Tensor, shard_id=None, **_ignored) -> None:
    if shard_id in param.shards:
        raise ValueError(f"EXL3 tensor for shard {shard_id!r} was loaded twice")
    param.shards[shard_id] = loaded_weight.to(param.target_device, copy=True)


class Exl3Collector(BasevLLMParameter):
    def __new__(cls, target_device, **kwargs):
        return super().__new__(cls, torch.empty(0, dtype=torch.uint8), **kwargs)

    def __init__(self, target_device):
        super().__init__(data=self.data, weight_loader=_collect)
        self.shards: dict = {}
        self.target_device = target_device

    # vLLM's v2 layer loaders call these on the parameter; all of them mean "here is one shard".
    def load_column_parallel_weight(self, loaded_weight, **kw):
        _collect(self, loaded_weight)

    load_row_parallel_weight = load_column_parallel_weight

    def load_merged_column_weight(self, loaded_weight, shard_id=None, **kw):
        _collect(self, loaded_weight, shard_id)

    load_qkv_weight = load_merged_column_weight
