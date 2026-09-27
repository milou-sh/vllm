# kernels/hopper — EXL3 trellis kernels for sm_90 (H100 / H200)

Three torch extensions built by `setup.py` (`./build.sh` on a box with CUDA 13.0 + torch 2.13; the Docker image builds all three):

| module | sources | what |
|---|---|---|
| `aikido_exl3_kernels` | `exl3_hopper.cu`, `exl3_hopper_template.h`, `exl3_hopper_inst.h`, `exl3_hopper_kernels.h`, `exl3_decode.cuh`, `exl3_had.cuh` | dense linears: vLLM's Marlin GEMM template with the int4 dequant replaced by ExLlamaV3's window extraction + MUL1/MCG decode. K=4 and K=6 (lm_head), rows 1-64 per launch (more rows split on the host), fused groups (q/k/v, gate/up, GDN qkv/z) in one launch through a shard map, bf16 in/out, output Hadamard inside the launch, slot reduce for the rows<=8 family, batched input Hadamard + strided output store for the many-row (prefill) path. |
| `aikido_exl3_moe_kernels` | `exl3_hopper_moe.cu`, `exl3_hopper_moe_template.h`, `exl3_hopper_moe_inst.h`, `exl3_hopper_moe_kernels.h`, `exl3_moe_orig_builder.cu` | routed experts: vLLM's Marlin-MoE structure with the EXL3 trellis decode; experts stacked `[E, ...]`, routing via `moe_align_block_size`, five launches per layer, automatic launch geometry. K=3 and K=4. `exl3_moe_orig_builder.cu` writes the original-basis weights (`W = diag(suh) H W_hat H diag(svh)`, ExLlamaV3's arithmetic) of every expert transposed straight into vLLM's `[E, out, in]` layout for the opt-in large-batch expert tier. |
| `aikido_exl3_moe_wgmma` | `exl3_moe_wgmma.cu` | sm_90a WGMMA prefill path for routed experts. |

Load-time repack of the trellis into the lane-major layout is a pure, invertible permutation; decoded weights are bit-identical to
`exllamav3_ext.reconstruct`. The source revision's parity suite checks dense, MoE, stress, and original-basis paths.

Build knobs (`setup.py`): `AIKIDO_MBS` (row-block families), `AIKIDO_CODEBOOKS` (0 = 3INST, 1 = MCG, 2 = MUL1), `AIKIDO_KBITS`, `AIKIDO_MOE`
(`1` default, `0` skip, `only`), `AIKIDO_MOE_MBS`, `AIKIDO_PROBES=1` (timing probes, never served).
Runtime switches (plugin): `AIKIDO_EXL3_HOPPER=1` (dense kernels), `AIKIDO_EXL3_HOPPER_MOE=1` (expert kernels), `AIKIDO_EXL3_MOE_ORIG=1`
(opt-in original-basis large-batch expert tier, see `models/qwen3.6-35b-a3b/README.md`), `AIKIDO_EXL3_MOE_ORIG_CACHE_GB` (GiB of
RESIDENT tier layers, default 0 = all transient, one shared arena), `AIKIDO_EXL3_MOE_ORIG_BUILDER` (`exl3` = ExLlamaV3's W bit for bit,
`fp32` = one final rounding, resident only), `AIKIDO_EXL3_MOE_ORIG_MIN_TOKENS` / `AIKIDO_EXL3_MOE_ORIG_TRANSIENT_MIN_TOKENS`
(measured crossovers with the trellis kernel, 512 / 4096). Kernel-level tuning knobs reachable through the raw extension:
`set_blocks_per_sm` (dense; 2 co-resident blocks per SM measure 7.96 ms per step at rows 1-8 vs 8.36 with 1, but lose at rows 16),
`moe_set_blocks_per_sm`, `set_had_warps`, `set_in_had_inlaunch`, `set_out_had_inlaunch`, `set_probes`.

Third-party code and licences: `third_party/` and the repo `NOTICE`.
