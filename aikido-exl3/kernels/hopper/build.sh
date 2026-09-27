#!/usr/bin/env bash
# Build on the H200 box: ./build.sh [small|probe]   (log: build.log, ends with BUILD_OK or BUILD_FAILED)
cd "$(dirname "$0")"; source /scratch/aikido/env.sh
[ "$1" = small ] && export AIKIDO_MBS=0,1 AIKIDO_CODEBOOKS=2
[ "$1" = moe ] && export AIKIDO_MOE=only          # MoE front: only the aikido_exl3_moe_kernels extension (setup.py)
[ "$2" = shuffle ] && export AIKIDO_WRAP_LOAD=0    # A/B: K=4 wrap bits by lane shuffle instead of the second load
[ "$2" = forceout ] && export AIKIDO_FORCE_OUT_FLAGS=1
[ "$1" = probe ] && export AIKIDO_PROBES=1 AIKIDO_MBS=0,1 AIKIDO_CODEBOOKS=${3:-2,3,4,5} AIKIDO_KBITS=4   # timing probes (exl3_decode.cuh), never served
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=${AIKIDO_JOBS:-${MAX_JOBS:-12}}
if nice -n 19 python setup.py build_ext --build-lib build/lib > build.log 2>&1; then echo BUILD_OK >> build.log; else echo BUILD_FAILED >> build.log; fi
