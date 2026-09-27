// AIKIDO: programmatic dependent launch (Hopper, sm_90). A kernel launched with
// cudaLaunchAttributeProgrammaticStreamSerialization may begin before the previous kernel in its stream (its
// "primary") has finished: griddepcontrol.wait blocks until every prerequisite grid has completed and its writes
// are visible; griddepcontrol.launch_dependents lets the next kernel in the stream start launching (its blocks
// then run their own prologue while this grid finishes). Both are no-ops for a kernel launched without the
// attribute. Invariant used by every kernel of the MoE chain: EVERY block executes the wait before it exits, so
// the completion of a grid implies the completion of its primary; a kernel's pre-wait prologue may therefore read
// anything produced by grids before its primary (routing tables), and must not touch what the primary reads or
// writes. Pure scheduling: no arithmetic changes, bit-identical results.
#pragma once

__device__ __forceinline__ void aikido_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void aikido_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// AIKIDO PDL per-slot kernel argument: bit 0 wait (launch carries the attribute), bit 1 trigger in every block after
// the wait, bit 3 trigger only in the last eighth of the grid (linear block id): the dependent can launch once every
// block of this grid has triggered or exited, so only the tail wave pays the serialized trigger.
// bit 4 (16): trigger at the very top, before the wait (align_decode: one block, so the next launch - had_in, which
// does not read align's output - may run concurrently; had_in then waits at its end, see bit 2 of had_in).
__device__ __forceinline__ void aikido_pdl_trig(int pdl) {
  if (pdl & 2) aikido_pdl_trigger();
  else if (pdl & 8) {
    const unsigned total = gridDim.x * gridDim.y * gridDim.z;
    const unsigned lin = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (lin >= total - total / 8) aikido_pdl_trigger();
  }
}
__device__ __forceinline__ void aikido_pdl_slot(int pdl) {
  if (pdl & 16) aikido_pdl_trigger();
  if (pdl & 1) aikido_pdl_wait();
  if (!(pdl & 16)) aikido_pdl_trig(pdl);
}
