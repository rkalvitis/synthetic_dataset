"""Minimal reader for 3D Gaussian Splatting PLY files (no plyfile dependency)."""
from __future__ import annotations

import numpy as np

_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


def read_gaussian_ply(path: str) -> dict[str, np.ndarray]:
    """Return a dict {property_name: (N,) array} for the 'vertex' element of a 3DGS PLY."""
    with open(path, "rb") as f:
        header: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header found")
            line = line.decode("ascii", errors="replace").strip()
            header.append(line)
            if line == "end_header":
                break

        if header[0] != "ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = next(l for l in header if l.startswith("format")).split()[1]
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"{path}: only binary PLY is supported, got {fmt}")
        endian = "<" if fmt == "binary_little_endian" else ">"

        # Only the first element (vertex) is read; later elements are ignored.
        n = None
        props: list[tuple[str, str]] = []
        for l in header:
            if l.startswith("element"):
                if n is not None:
                    break
                _, name, count = l.split()
                if name != "vertex":
                    raise ValueError(f"{path}: first element must be 'vertex', got {name}")
                n = int(count)
            elif l.startswith("property") and n is not None:
                parts = l.split()
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported")
                props.append((parts[2], endian + _PLY_TYPES[parts[1]]))

        dtype = np.dtype(props)
        raw = f.read(n * dtype.itemsize)
        if len(raw) != n * dtype.itemsize:
            raise ValueError(f"{path}: truncated file")
        arr = np.frombuffer(raw, dtype=dtype)
    return {name: np.ascontiguousarray(arr[name]) for name, _ in props}
