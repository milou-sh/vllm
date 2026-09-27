import json
import struct

import pytest

from helpers import exl3_linear, write_safetensors

from aikido_exl3.format import FormatError, load_manifest
from aikido_exl3.format.safetensors_header import read_header, read_headers
from aikido_exl3.format.tr3 import load_tr3


def test_rejects_large_single_header(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 64 * 1024 * 1024 + 1))
    with pytest.raises(FormatError, match="implausible safetensors header length"):
        read_header(path)


def test_rejects_large_aggregate_headers(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"model-{index}.safetensors"
        with path.open("wb") as shard:
            shard.write(struct.pack("<Q", 64 * 1024 * 1024))
            shard.truncate(64 * 1024 * 1024 + 8)
        paths.append(str(path))
    with pytest.raises(FormatError, match="checkpoint limit"):
        read_headers(paths)


def test_manifest_still_reads_normal_headers(tmp_path):
    write_safetensors(tmp_path / "model.safetensors", exl3_linear("a.proj", 128, 128, 48, marker="mcg"))
    manifest = load_manifest(tmp_path)
    assert manifest.matrices["a.proj"].bits.value == 3


def test_tier_bitmap_cannot_escape_checkpoint(tmp_path):
    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        k, n = (128, 256) if proj == "down_proj" else (256, 128)
        tensors.update(exl3_linear(f"model.layers.0.mlp.experts.0.{proj}.rank0", k, n, 48, marker="mcg"))
    write_safetensors(tmp_path / "model.safetensors", tensors)
    (tmp_path / "config.json").write_text(json.dumps({
        "hybrid_tr3_tail": {"format": "exl3-trellis", "tp": 1, "tier_bitmap": "../tier_bitmap.json"}
    }))
    with pytest.raises(FormatError, match="escapes the checkpoint directory"):
        load_tr3(tmp_path)


def test_tier_bitmap_must_be_relative(tmp_path):
    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        k, n = (128, 256) if proj == "down_proj" else (256, 128)
        tensors.update(exl3_linear(f"model.layers.0.mlp.experts.0.{proj}.rank0", k, n, 48, marker="mcg"))
    write_safetensors(tmp_path / "model.safetensors", tensors)
    (tmp_path / "config.json").write_text(json.dumps({
        "hybrid_tr3_tail": {"format": "exl3-trellis", "tp": 1, "tier_bitmap": "/etc/passwd"}
    }))
    with pytest.raises(FormatError, match="must be checkpoint-relative"):
        load_tr3(tmp_path)
