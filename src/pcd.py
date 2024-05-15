import os
import struct

import lzf
import numpy as np

# Reference:
# https://pointclouds.org/documentation/tutorials/pcd_file_format.html

_header = [
    "VERSION",
    "FIELDS",
    "SIZE",
    "TYPE",
    "COUNT",
    "WIDTH",
    "HEIGHT",
    "VIEWPOINT",
    "POINTS",
    "DATA",
]


def parse_rgb32(x):
    if not isinstance(x, np.ndarray) or x.dtype != np.float32:
        raise ValueError("x must be a float32 numpy array.")

    R = np.ndarray((len(x), 4), buffer=x.tobytes(), dtype=np.uint8)
    return R


def parse_header(fd):
    H = {}

    while len(H) != len(_header):
        line = fd.readline()
        vals = line.decode("ascii").split()

        if vals[0] == "#":
            continue

        H[vals[0]] = vals[1:]

        if vals[0] == "DATA":
            break

    # assume all fields are parsed
    H["VERSION"] = H["VERSION"][0]
    H["WIDTH"] = int(H["WIDTH"][0])
    H["HEIGHT"] = int(H["HEIGHT"][0])
    H["POINTS"] = int(H["POINTS"][0])
    H["DATA"] = H["DATA"][0]

    return H


def parse_header_by_file(f, return_size=False):
    if not isinstance(f, str):
        raise TypeError("f should be a string.")

    if not os.path.exists(f):
        raise OSError("%s not found" % f)

    with open(f, "rb") as fd:
        header = parse_header(fd)
        size = fd.tell()

    if return_size:
        return header, size
    return header


def parse_binary_data(data, shape, dtype):
    assert len(data) == dtype.itemsize * np.prod(shape)
    return np.ndarray(shape, buffer=data, dtype=dtype)


def parse_binary_compressed_data(data, shape, dtype):
    fmt = "<II"

    compressed_size, uncompressed_size = struct.unpack(fmt, data[:8])

    data = data[8:]
    data = data[:compressed_size]
    data = lzf.decompress(data, uncompressed_size)

    B = np.empty(shape, dtype=dtype)
    for n, t in dtype.descr:
        size = np.dtype(t).itemsize * np.prod(shape)
        buf = data[:size]
        data = data[size:]

        B[n] = np.frombuffer(buf, dtype=t).reshape(shape)

    return B


_loader = {
    "binary": parse_binary_data,
    "binary_compressed": parse_binary_compressed_data,
}


def form_dtype_by_header(H):
    # form datatypes
    assert (
        len(H["FIELDS"]) == len(H["SIZE"]) == len(H["TYPE"]) == len(H["COUNT"])
    )

    assert H["WIDTH"] * H["HEIGHT"] == H["POINTS"]

    dtype = []
    for field, size, t, cnt in zip(
        H["FIELDS"], H["SIZE"], H["TYPE"], H["COUNT"]
    ):

        dt = (field, t.lower() + size)

        cnt = int(cnt)
        if cnt > 1:
            dt = dt + ((cnt,))
        dtype.append(dt)
    dtype = np.dtype(dtype)

    return dtype


def read(f, return_header=False):

    H, size = parse_header_by_file(f, return_size=True)

    dtype = form_dtype_by_header(H)
    shape = (H["HEIGHT"], H["WIDTH"]) if H["HEIGHT"] > 1 else H["WIDTH"]

    with open(f, "rb") as fd:
        fd.seek(size)

        if H["DATA"].lower() == "ascii":
            B = np.loadtxt(fd, dtype=dtype, delimiter=" ")
        else:
            loader = _loader[H["DATA"].lower()]
            B = loader(fd.read(), shape, dtype)

    if return_header:
        return B, H
    return B


def write_ascii_data(x):
    raise NotImplementedError


def write_binary_data(x):
    raise NotImplementedError


def write_binary_compressed_data(x):
    raise NotImplementedError


def write(f, x):
    raise NotImplementedError
