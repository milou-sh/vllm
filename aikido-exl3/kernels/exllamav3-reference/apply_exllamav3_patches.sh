#!/usr/bin/env bash
# Usage: apply_exllamav3_patches.sh <exllamav3 source tree>
set -euo pipefail
tree="${1:?exllamav3 source tree}"
f="$tree/exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh"
old='#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)$'
new='#if defined(__CUDA_ARCH__) \&\& (__CUDA_ARCH__ > 890) \&\& defined(EXL3_SM90_BARRIER)'
n=$(grep -c "$old" "$f" || true)
if [[ "$n" != 3 ]]; then
  echo "expected 3 sm_90 barrier gates in $f, found $n: upstream changed, re-audit before building" >&2
  exit 1
fi
sed -i.orig "s/$old/$new/" "$f"
[[ $(grep -c 'defined(EXL3_SM90_BARRIER)' "$f") == 3 ]]
echo "patched: 3 barrier gates in $(basename "$f")"
