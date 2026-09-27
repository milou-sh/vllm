# aikido-exl3

EXL3 quantization plugin and Hopper (sm_90) kernels for this vLLM v0.29.0 line. It serves
`davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw` (revision `6d6bd738`) on 4x H200 at TP4.

Source revision `3ffdb5d8f8f9136fefb2ba16dd2e32fcc3e6346a` plus the checkpoint-validation patchset: bounded
checkpoint header parsing, checkpoint-relative metadata paths, and MoE tensor validation against the header scan and
vLLM layer dimensions before native kernels run.

- `shared/python`: the `aikido_exl3` vLLM plugin (`quant_method: "exl3"`).
- `kernels/hopper`: dense, MoE, and WGMMA prefill kernels.
- `kernels/exllamav3-reference`: patches for the ExLlamaV3 `958ec933` reference extension.

Everything is off by default. `AIKIDO_EXL3_HOPPER_MOE` and the `AIKIDO_NONMOE_*` switches enable the kernels.

Decoded weights must stay bit-identical to ExLlamaV3 `reconstruct`; that parity is the gate for every kernel change.
Third-party code and licences are listed in `NOTICE`.

Aikido Box builds its vLLM image from a pinned commit of this directory; see `docker/vllm` in `AikidoSec/aikido-box`.
