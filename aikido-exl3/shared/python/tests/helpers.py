import json
import struct


DTYPE_BYTES = {"I16": 2, "I32": 4, "F16": 2, "BF16": 2, "F32": 4}


def write_safetensors(path, tensors, metadata=None):
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = DTYPE_BYTES[dtype]
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def exl3_linear(key, k, n, words, marker="mul1"):
    tensors = {
        f"{key}.trellis": ("I16", (k // 16, n // 16, words)),
        f"{key}.suh": ("F16", (k,)),
        f"{key}.svh": ("F16", (n,)),
    }
    if marker:
        tensors[f"{key}.{marker}"] = ("I32", ())
    return tensors
