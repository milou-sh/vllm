# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    nvfp4_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_fi_or_cutlass,
    prepare_nvfp4_moe_layer_for_flashinfer_cutedsl,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    FlashinferMoeBackend,
    get_flashinfer_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    kE2M1ToFloat_handle,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    FP8_COMPUTE_BLOCK_SIZE,
    NVFP4_GROUP_SIZE,
    _is_hopper_without_native_fp4,
    ensure_e2m1_table_on_device,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)

logger = init_logger(__name__)


class NvFp4MoeBackend(Enum):
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    FLASHINFER_CUTEDSL = "FLASHINFER_CUTEDSL"
    FLASHINFER_CUTEDSL_BATCHED = "FLASHINFER_CUTEDSL_BATCHED"
    FLASHINFER_B12X = "FLASHINFER_B12X"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    FP8_COMPUTE = "FP8_COMPUTE"
    MARLIN = "MARLIN"
    EMULATION = "EMULATION"


FLASHINFER_NVFP4_MOE_BACKENDS = [
    NvFp4MoeBackend.FLASHINFER_TRTLLM,
    NvFp4MoeBackend.FLASHINFER_CUTLASS,
    NvFp4MoeBackend.FLASHINFER_CUTEDSL,
    NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
    NvFp4MoeBackend.FLASHINFER_B12X,
]

fi_2_vllm_backend_map: dict[FlashinferMoeBackend, NvFp4MoeBackend] = {
    FlashinferMoeBackend.CUTLASS: NvFp4MoeBackend.FLASHINFER_CUTLASS,
    FlashinferMoeBackend.TENSORRT_LLM: NvFp4MoeBackend.FLASHINFER_TRTLLM,
    FlashinferMoeBackend.CUTEDSL: NvFp4MoeBackend.FLASHINFER_CUTEDSL,
}


def is_global_sf_supported_for_nvfp4_backend(backend: NvFp4MoeBackend) -> bool:
    # Checks whether `backend` supports quantizing with scaling factors
    # of all experts in Expert Parallel Mode when all experts are not
    # on the same rank.

    return backend in FLASHINFER_NVFP4_MOE_BACKENDS


def backend_to_kernel_cls(
    backend: NvFp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == NvFp4MoeBackend.FLASHINFER_TRTLLM:
        from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
            TrtLlmNvFp4ExpertsModular,
            TrtLlmNvFp4ExpertsMonolithic,
        )

        # NOTE: prefer Monolthic > Modular, so return Monolithic first.
        return [
            TrtLlmNvFp4ExpertsMonolithic,
            TrtLlmNvFp4ExpertsModular,
        ]

    elif backend == NvFp4MoeBackend.FLASHINFER_CUTLASS:
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (  # noqa: E501
            FlashInferExperts,
        )

        return [FlashInferExperts]

    elif backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL:
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (  # noqa: E501
            FlashInferCuteDSLExperts,
        )

        return [FlashInferCuteDSLExperts]

    elif backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED:
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe import (  # noqa: E501
            FlashInferCuteDSLBatchedExperts,
        )

        return [FlashInferCuteDSLBatchedExperts]

    elif backend == NvFp4MoeBackend.FLASHINFER_B12X:
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (  # noqa: E501
            FlashInferB12xExperts,
        )

        return [FlashInferB12xExperts]

    elif backend == NvFp4MoeBackend.VLLM_CUTLASS:
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            CutlassExpertsFp4,
        )

        return [CutlassExpertsFp4]

    elif backend == NvFp4MoeBackend.FP8_COMPUTE:
        from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
            TritonOrDeepGemmExperts,
        )

        return [TritonOrDeepGemmExperts]

    elif backend == NvFp4MoeBackend.MARLIN:
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            MarlinExperts,
        )

        return [MarlinExperts]
    elif backend == NvFp4MoeBackend.EMULATION:
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
            Nvfp4QuantizationEmulationTritonExperts,
        )

        return [Nvfp4QuantizationEmulationTritonExperts]
    else:
        raise ValueError(f"Unknown NvFP4 MoE backend: {backend.value}")


def map_nvfp4_backend(runner_backend: MoEBackend) -> NvFp4MoeBackend:
    """Map user's MoEBackend to NvFp4MoeBackend."""
    mapping = {
        "cutlass": NvFp4MoeBackend.VLLM_CUTLASS,
        "flashinfer_trtllm": NvFp4MoeBackend.FLASHINFER_TRTLLM,
        "flashinfer_cutlass": NvFp4MoeBackend.FLASHINFER_CUTLASS,
        "flashinfer_cutedsl": NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        "flashinfer_b12x": NvFp4MoeBackend.FLASHINFER_B12X,
        "fp8_compute": NvFp4MoeBackend.FP8_COMPUTE,
        "marlin": NvFp4MoeBackend.MARLIN,
        "emulation": NvFp4MoeBackend.EMULATION,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for NvFP4 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_nvfp4_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[NvFp4MoeBackend, type[mk.FusedMoEExperts]]:
    """
    Select the primary NvFP4 MoE backend
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    # NOTE: the kernels are selected in the following order.
    # FLASHINFER_B12X is intentionally excluded from auto-selection until
    # the upstream CUTLASS SM121 MMA op guard is resolved; use
    # moe_backend="flashinfer_b12x" to opt in explicitly.
    AVAILABLE_BACKENDS = [
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.VLLM_CUTLASS,
        NvFp4MoeBackend.FP8_COMPUTE,
        NvFp4MoeBackend.MARLIN,
        NvFp4MoeBackend.EMULATION,
    ]

    NVFP4_BACKENDS_WITH_CLAMP = {
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
    }

    if config.swiglu_limit is not None:
        AVAILABLE_BACKENDS = [
            b for b in AVAILABLE_BACKENDS if b in NVFP4_BACKENDS_WITH_CLAMP
        ]

    use_batched = config.moe_parallel_config.use_batched_activation_format
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if use_batched
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: NvFp4MoeBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using '{backend.value}' NvFp4 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: NvFp4MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"NvFp4 MoE backend '{backend.value}' does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"NvFp4 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: NvFp4MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[NvFp4MoeBackend, type[mk.FusedMoEExperts]]:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls

        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_nvfp4_backend(runner_backend)
        # For batched activation format, use batched variant if available.
        if (
            activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and requested_backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL
        ):
            requested_backend = NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED
        if (
            config.swiglu_limit is not None
            and requested_backend not in NVFP4_BACKENDS_WITH_CLAMP
        ):
            raise ValueError(
                f"Model sets swiglu_limit={config.swiglu_limit}, but the "
                f"explicitly requested moe_backend={runner_backend!r} does "
                f"not apply the SwiGLU clamp. Use 'flashinfer_trtllm' or "
                f"'flashinfer_cutlass' instead."
            )
        return _return_or_raise(
            requested_backend, config, weight_key, activation_key, activation_format
        )

    if envs.is_set("VLLM_USE_FLASHINFER_MOE_FP4"):
        if not envs.VLLM_USE_FLASHINFER_MOE_FP4:
            # If the user rejects FlashInfer remove those backends.
            for b in FLASHINFER_NVFP4_MOE_BACKENDS:
                if b in AVAILABLE_BACKENDS:
                    AVAILABLE_BACKENDS.remove(b)

        elif envs.is_set("VLLM_FLASHINFER_MOE_BACKEND"):
            # If user is explicit about backend, validate it.
            backend = fi_2_vllm_backend_map[get_flashinfer_moe_backend()]
            if (
                config.swiglu_limit is not None
                and backend not in NVFP4_BACKENDS_WITH_CLAMP
            ):
                raise ValueError(
                    f"Model sets swiglu_limit={config.swiglu_limit}, but the "
                    f"FlashInfer backend selected via VLLM_FLASHINFER_MOE_BACKEND "
                    f"({backend.value}) does not apply the SwiGLU clamp."
                )
            return _return_or_raise(
                backend, config, weight_key, activation_key, activation_format
            )
        else:
            # If the user is not explicit about the backend, try each.
            fi_backends = [
                b
                for b in FLASHINFER_NVFP4_MOE_BACKENDS
                if config.swiglu_limit is None or b in NVFP4_BACKENDS_WITH_CLAMP
            ]
            for backend in fi_backends:
                for k_cls in backend_to_kernel_cls(backend):
                    supported, reason = k_cls.is_supported_config(
                        k_cls,
                        config,
                        weight_key,
                        activation_key,
                        activation_format,
                    )
                    if supported:
                        logger.info_once(_make_log_backend(backend))
                        return backend, k_cls
                    else:
                        logger.debug_once(_make_log_unsupported(backend, reason))

            raise NotImplementedError(
                "Found VLLM_USE_FLASHINFER_MOE_FP4=1, but no "
                "FlashInfer NVFP4 MoE backend supports the configuration."
            )

    if envs.VLLM_TEST_FORCE_FP8_MARLIN:
        backend = NvFp4MoeBackend.MARLIN
        return _return_or_raise(
            backend, config, weight_key, activation_key, activation_format
        )

    # On Hopper without native FP4 the only remaining candidate is Marlin, which
    # dequantizes FP4->FP16 and computes on FP16 tensor cores. Converting the
    # expert weights to FP8 once at load time is ~4x faster on paper, so prefer
    # it. VLLM_NVFP4_GEMM_BACKEND is honoured: set it to "marlin" to opt out.
    # Skipped when the deployment needs a SwiGLU clamp (FP8_COMPUTE does not
    # apply one) or a batched activation format (TritonOrDeepGemmExperts is
    # Standard-format only).
    if (
        NvFp4MoeBackend.FP8_COMPUTE in AVAILABLE_BACKENDS
        and activation_format == mk.FusedMoEActivationFormat.Standard
        and config.swiglu_limit is None
        and _is_hopper_without_native_fp4()
        and envs.VLLM_NVFP4_GEMM_BACKEND in (None, "fp8-compute")
    ):
        from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
            TritonOrDeepGemmExperts,
        )

        logger.info_once(
            "Using FP8_COMPUTE NvFp4 MoE backend on Hopper "
            "(converting NVFP4 expert weights to FP8)."
        )
        return NvFp4MoeBackend.FP8_COMPUTE, TritonOrDeepGemmExperts

    # Select kernels in order of backend.
    for backend in AVAILABLE_BACKENDS:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No NvFp4 MoE backend supports the deployment configuration."
    )


def _convert_nvfp4_moe_to_fp8_compute(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    None,
    None,
    torch.Tensor,
    torch.Tensor,
    None,
    None,
]:
    """Convert NVFP4 MoE expert weights to FP8 block-quantized format.

    Dequantizes each expert's FP4 weights to BF16, then re-quantizes to FP8
    with [128, 128] block scaling for the DeepGEMM / Triton FP8 MoE kernels.
    Done per expert to bound the transient BF16 allocation to one expert.
    """
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    logger.info_once(
        "Converting NVFP4 MoE expert weights to FP8 for native Hopper "
        "tensor core compute."
    )

    ensure_e2m1_table_on_device(w13.device)

    def _convert_experts(
        weights: torch.Tensor,
        scales: torch.Tensor,
        global_scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert (E, N, K/2) packed FP4 -> (E, N, K) FP8 + block scales."""
        fp8_list, scale_list = [], []
        for e in range(weights.shape[0]):
            gs = global_scales[e] if global_scales.dim() > 0 else global_scales
            # swizzle=False: expert scales are still in the checkpoint's linear
            # layout here, since this branch replaces the kernel-specific
            # swizzling prepare step rather than running after it.
            w_bf16 = dequantize_to_dtype(
                weights[e].view(torch.uint8),
                scales[e],
                gs,
                torch.bfloat16,
                block_size=NVFP4_GROUP_SIZE,
                swizzle=False,
            )
            w_fp8, w_scale = per_block_cast_to_fp8(w_bf16, FP8_COMPUTE_BLOCK_SIZE)
            fp8_list.append(w_fp8)
            scale_list.append(w_scale)
            del w_bf16
        return torch.stack(fp8_list), torch.stack(scale_list)

    w13_fp8, w13_fp8_scale = _convert_experts(w13, w13_scale, w13_scale_2)
    w2_fp8, w2_fp8_scale = _convert_experts(w2, w2_scale, w2_scale_2)

    return (w13_fp8, w13_fp8_scale, None, None, w2_fp8, w2_fp8_scale, None, None)


def convert_to_nvfp4_moe_kernel_format(
    nvfp4_backend: NvFp4MoeBackend,
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor | None,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor | None,
    is_act_and_mul: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if nvfp4_backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL:
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = prepare_nvfp4_moe_layer_for_flashinfer_cutedsl(
            layer=layer,
            w13=w13,
            w13_scale=w13_scale,
            w13_scale_2=w13_scale_2,
            a13_scale=a13_scale,
            w2=w2,
            w2_scale=w2_scale,
            w2_scale_2=w2_scale_2,
            a2_scale=a2_scale,
        )
    elif (
        nvfp4_backend in FLASHINFER_NVFP4_MOE_BACKENDS
        or nvfp4_backend == NvFp4MoeBackend.VLLM_CUTLASS
    ):
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
            backend=nvfp4_backend,
            layer=layer,
            w13=w13,
            w13_scale=w13_scale,
            w13_scale_2=w13_scale_2,
            a13_scale=a13_scale,
            w2=w2,
            w2_scale=w2_scale,
            w2_scale_2=w2_scale_2,
            a2_scale=a2_scale,
            is_act_and_mul=is_act_and_mul,
        )
    elif nvfp4_backend == NvFp4MoeBackend.FP8_COMPUTE:
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = _convert_nvfp4_moe_to_fp8_compute(
            w13,
            w13_scale,
            w13_scale_2,
            w2,
            w2_scale,
            w2_scale_2,
        )
    elif nvfp4_backend == NvFp4MoeBackend.MARLIN:
        a13_scale = None
        a2_scale = None
        (
            w13,
            w13_scale,
            w13_scale_2,
            w2,
            w2_scale,
            w2_scale_2,
        ) = prepare_nvfp4_moe_layer_for_marlin(
            layer=layer,
            w13=w13,
            w13_scale=w13_scale,
            w13_scale_2=w13_scale_2,
            w2=w2,
            w2_scale=w2_scale,
            w2_scale_2=w2_scale_2,
            is_act_and_mul=is_act_and_mul,
        )
    elif nvfp4_backend == NvFp4MoeBackend.EMULATION:
        # Move the E2M1 lookup table to the device now, because
        # `.to(device)` is not allowed during CUDA graph capture.
        kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(w13.device)

        if a13_scale is None or a2_scale is None:
            raise ValueError(
                "Activation global scales should not be None, got"
                f" a13_scale={a13_scale}, a2_scale={a2_scale}"
            )

        if torch.unique(a13_scale).numel() != 1 or torch.unique(a2_scale).numel() != 1:
            logger.warning_once(
                "In NVFP4 linear, the activation global scale for inputs are different"
                " for MOE w13 (gate_up_proj) layer or MOE w2 (down_proj). Using"
                " a13_scale = a13_scale.max() and a2_scale = a2_scale.max()."
            )

        # 1. We take the max following e.g. quantization/utils/flashinfer_fp4_moe.py.
        # 2. moe_kernel_quantize_input -> ref_nvfp4_quant_dequant
        # use the inverse scale directly (large global scale).
        # NOTE: Before this point, `a13_scale` and `a2_scale` are such that:
        # `FP8_MAX = activation[expert_id].abs().max() * global_scale[expert_id]`,
        # and `global_scale[expert_id]` are small (~1e-4).
        # Taking the largest global scale likely results in overflowing the FP8 range
        # for other experts - other selection strategies may be used.
        a13_scale = 1.0 / a13_scale.max().to(torch.float32)
        a2_scale = 1.0 / a2_scale.max().to(torch.float32)
    else:
        raise ValueError(f"Unknown NvFp4 backend for MoE: {nvfp4_backend}")

    return (
        w13,
        w13_scale,
        w13_scale_2,
        a13_scale,
        w2,
        w2_scale,
        w2_scale_2,
        a2_scale,
    )


def make_nvfp4_moe_quant_config(
    backend: NvFp4MoeBackend,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    swiglu_limit: float | None = None,
) -> FusedMoEQuantConfig:
    if backend == NvFp4MoeBackend.FP8_COMPUTE:
        # Weights are plain block-scaled FP8 after conversion, so this is an
        # ordinary FP8 w8a8 config; the NVFP4 global scales are already folded
        # into the dequantized values and must not be applied again.
        return fp8_w8a8_moe_quant_config(
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=FP8_COMPUTE_BLOCK_SIZE,
        )

    if backend == NvFp4MoeBackend.MARLIN:
        return nvfp4_w4a16_moe_quant_config(
            g1_alphas=w13_scale_2,
            g2_alphas=w2_scale_2,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
        )
    elif backend == NvFp4MoeBackend.EMULATION:
        return nvfp4_moe_quant_config(
            g1_alphas=w13_scale_2,
            g2_alphas=w2_scale_2,
            a1_gscale=a13_scale,
            a2_gscale=a2_scale,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            gemm1_clamp_limit=swiglu_limit,
        )

    # Pass w13_scale_2 / w2_scale_2 directly as g1/g2_alphas.
    # The expert's process_weights_after_loading will fuse activation
    # scales in-place. Since the quant config references the same tensor
    # as the registered parameter, EPLB rearrangement stays in sync.
    return nvfp4_moe_quant_config(
        g1_alphas=w13_scale_2,
        g2_alphas=w2_scale_2,
        a1_gscale=(1.0 / a13_scale),
        a2_gscale=(1.0 / a2_scale),
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        # NOTE(rob): this is a hack until the MoE kernels
        # create their own quant configs. TRTLLM kernel
        # does not accept swizzled input quant scales.
        is_scale_swizzled=(
            backend
            not in (
                NvFp4MoeBackend.FLASHINFER_TRTLLM,
                NvFp4MoeBackend.FLASHINFER_CUTEDSL,
            )
        ),
        gemm1_clamp_limit=swiglu_limit,
    )


def make_nvfp4_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=issubclass(experts_cls, mk.FusedMoEExpertsMonolithic),
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )

    return kernel
