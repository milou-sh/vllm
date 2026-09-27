"""EXL3 (ExLlamaV3 trellis format) for stock vLLM. See docs/ARCHITECTURE.md.

Layers, lowest first; a layer never imports from one above it
(tests/test_layering.py):  format < arch < kernels < runtime < vllm_glue < parity, tools
"""
__version__ = "0.0.1"
