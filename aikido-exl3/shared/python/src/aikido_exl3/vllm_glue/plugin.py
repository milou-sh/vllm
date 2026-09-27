"""vllm.general_plugins entry point. Runs in every vLLM process before the model is resolved.
Must be idempotent and must not create a CUDA context."""
_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    from . import compat
    from .config import Exl3Config
    compat.register_quantization_config("exl3")(Exl3Config)
    from ..runtime import ops  # noqa: F401  defines the custom ops once, outside any torch.compile trace
    from . import skinny   # opt-in skinny bf16 GEMM for the non-MoE linears (AIKIDO_NONMOE_SKINNY=1)
    skinny.install()
    from . import attn_reuse   # opt-in: sparse-MLA index conversion computed once per fresh top-k (AIKIDO_NONMOE_ATTN_REUSE=1)
    attn_reuse.install()
    _registered = True
