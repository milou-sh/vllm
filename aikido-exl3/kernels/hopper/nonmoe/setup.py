"""Build `aikido_nonmoe_kernels` (skinny bf16 GEMM for the non-MoE decode GEMMs, sm_90).
    TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --build-lib <out>"""
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="aikido_nonmoe_kernels",
    version="0.1.0",
    ext_modules=[cpp_extension.CUDAExtension(
        "aikido_nonmoe_kernels", ["skinny_gemm.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo", "-std=c++17", "--use_fast_math", "-Xptxas=-v"]})],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
