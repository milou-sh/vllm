# vLLM 0.29.0 with the aikido-exl3 plugin and Hopper kernels. Build context: this directory.
FROM vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b

ARG CUDA_ARCH="9.0"
ARG CUDA_PYTHON_INCLUDE=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
ARG EXLLAMAV3_ARCHIVE_SHA256=d5c52142cc2330b3d75ab25daee651c70baff87d44fdd9756ae8d2295f95112e
ARG EXLLAMAV3_COMMIT=958ec933361b24eb8426ec7222e5b0062a679dcd
ARG MAX_JOBS=12

COPY kernels/exllamav3-reference /opt/aikido/kernels/exllamav3-reference
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz', '/tmp/exllamav3.tar.gz')" \
    && echo "${EXLLAMAV3_ARCHIVE_SHA256}  /tmp/exllamav3.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/src/exllamav3 /opt/aikido/lib \
    && tar -xzf /tmp/exllamav3.tar.gz --strip-components=1 -C /opt/src/exllamav3 \
    && bash /opt/aikido/kernels/exllamav3-reference/apply_exllamav3_patches.sh /opt/src/exllamav3 \
    && cd /opt/src/exllamav3 \
    && CPATH="${CUDA_PYTHON_INCLUDE}" TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" MAX_JOBS="${MAX_JOBS}" python3 setup.py build_ext --build-lib /opt/aikido/lib \
    && rm -rf /opt/src/exllamav3 /tmp/exllamav3.tar.gz

COPY kernels/hopper /opt/aikido/kernels/hopper
RUN cd /opt/aikido/kernels/hopper \
    && CPATH="${CUDA_PYTHON_INCLUDE}" TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" MAX_JOBS="${MAX_JOBS}" python3 setup.py build_ext --build-lib /opt/aikido/lib \
    && rm -rf build \
    && cd nonmoe \
    && CPATH="${CUDA_PYTHON_INCLUDE}" TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" MAX_JOBS="${MAX_JOBS}" python3 setup.py build_ext --build-lib /opt/aikido/lib \
    && rm -rf build

COPY shared/python /opt/aikido/shared/python
COPY NOTICE /opt/aikido/NOTICE
RUN python3 -m pip install --disable-pip-version-check --no-build-isolation --no-deps -e /opt/aikido/shared/python

ENV PYTHONPATH=/opt/aikido/lib \
    AIKIDO_ROOT=/opt/aikido \
    AIKIDO_EXL3_HOPPER=0 \
    AIKIDO_EXL3_HOPPER_MOE=0 \
    AIKIDO_EXL3_MOE_ORIG=0 \
    AIKIDO_NONMOE_SKINNY=0 \
    AIKIDO_NONMOE_ATTN_REUSE=0 \
    AIKIDO_NONMOE_ATTN_SCHED=0 \
    AIKIDO_NONMOE_MERGE=0

LABEL org.opencontainers.image.title="vLLM 0.29.0 with aikido-exl3" \
      org.opencontainers.image.source="https://github.com/milou-sh/vllm"
