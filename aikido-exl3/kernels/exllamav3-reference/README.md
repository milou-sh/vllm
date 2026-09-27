# third_party

ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp) is the numerical reference for this project.

- Pin: `turboderp-org/exllamav3` tag `v1.5.1`.
- `apply_exllamav3_patches.sh <tree>` applies our build-time changes and fails if upstream moved:
  1. sm_90+ MGEMM `group_barrier` gated behind `EXL3_SM90_BARRIER` (default: `grid.sync()`), after
     yeasah/exllamav3 `dc1dc239` (MIT; original patch, made against v1.4.9, kept in `patches/` for reference).
     Upstream v1.5.1 still spins on a device-global barrier on sm_90+, which hangs under vLLM
     (research/06 section A6). No numeric effect: it only changes how cooperating threads synchronise.
- `build_on_box.sh` builds (a) an oracle venv from the upstream release wheel and (b) a patched sm_90 wheel
  against the vLLM venv's torch. Neither touches the serving venv.

## Code vendored into `csrc/` (the `aikido_exl3_kernels` extension, Hopper backend)

| path | origin | licence | changes |
|---|---|---|---|
| `csrc/exl3_hopper_template.h` | vLLM v0.29.0 `csrc/libtorch_stable/quantization/marlin/marlin_template.h` (Marlin, Copyright (C) 2024 Elias Frantar; modified by Neural Magic and the vLLM project) | Apache-2.0 | int4 dequant replaced by EXL3 trellis decode, extra `exl3_cb` template parameter, per-column scale removed; every change is marked `AIKIDO` |
| `csrc/third_party/marlin/{marlin.cuh,marlin_dtypes.cuh,marlin_mma.h,dequant.h,core/scalar_type.hpp,LICENSE}` | same vLLM tree (`csrc/libtorch_stable/quantization/marlin/`, `csrc/core/`) | Apache-2.0 | none |
| host launch logic in `csrc/exl3_hopper.cu` (`gemm_launch`, thread-config tables, shared-memory sizing) | follows vLLM `marlin.cu` | Apache-2.0 | reduced to the fp16 / 4-bit / no-scale case |
| `csrc/exl3_decode.cuh` | ExLlamaV3 v1.5.1 `exllamav3_ext/quant/exl3_gemv_kernel.cuh` (`dq8_regs_4bits`, `decode8`), `util.cuh` (`half4`, unions) | MIT, Copyright (c) 2025 Turboderp | template on the fragment type; arithmetic unchanged |
| `csrc/third_party/exllamav3/{codebook.cuh,hadamard_inner.cuh,LICENSE}`, `csrc/third_party/compat.cuh` | ExLlamaV3 v1.5.1 `exllamav3_ext/quant/`, `exllamav3_ext/compat.cuh` | MIT | none (`compat.cuh` sits one level up because `hadamard_inner.cuh` includes `../compat.cuh`) |
