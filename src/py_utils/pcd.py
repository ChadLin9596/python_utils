import os
import struct

import lzf
import numpy as np

from . import visualization_pptk

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

# this format is used for the visualization tool: cloud compare
# the fields 'x', 'y', 'z' are mandatory
# the data type is suggested to be float32 (float64 will raise an error)
# more fields can be added depending on the application
CLOUD_COMPARE_DTYPE = [
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("rgb", "<f4"),
]


def points_to_cloud_compare_pcd(
    xyz,
    intensity=None,
    s_min=None,
    s_max=None,
    color_map=None,
):

    # TODO: make intensity to intensity_or_rgb

    N = len(xyz)
    if np.shape(xyz) != (N, 3):
        raise ValueError("xyz should be (N, 3)")

    if intensity is not None and len(intensity) != N:
        raise ValueError("intensity should have length N")

    R = np.zeros(N, dtype=np.dtype(CLOUD_COMPARE_DTYPE))
    R["x"] = xyz[:, 0]
    R["y"] = xyz[:, 1]
    R["z"] = xyz[:, 2]

    if intensity is None:
        return R

    s_min = np.min(intensity) if s_min is None else s_min
    s_max = np.max(intensity) if s_max is None else s_max
    func = visualization_pptk.make_color(s_min, s_max, color_map=color_map)

    bgr = func(intensity)[:, ::-1]
    bgr = np.vstack([bgr.T, np.ones(N)]).T
    bgr = bgr * 255
    bgr = bgr.astype(np.uint8).tobytes()
    bgr = np.frombuffer(bgr, dtype=np.float32)

    R["rgb"] = bgr

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


def _write_header(fd, H):
    # currently only support binary

    fd.write(b"# .PCD v.7 - Point Cloud Data file format\n")
    fd.write(b"VERSION .7\n")
    fd.write(b"FIELDS %s\n" % b" ".join(H["FIELDS"]))
    fd.write(b"SIZE %s\n" % b" ".join(H["SIZE"]))
    fd.write(b"TYPE %s\n" % b" ".join(H["TYPE"]))
    fd.write(b"COUNT %s\n" % b" ".join(H["COUNT"]))
    fd.write(b"WIDTH %d\n" % H["WIDTH"])
    fd.write(b"HEIGHT %d\n" % H["HEIGHT"])
    fd.write(b"VIEWPOINT 0 0 0 1 0 0 0\n")
    fd.write(b"POINTS %d\n" % H["POINTS"])
    fd.write(b"DATA binary\n")


def write(f, x):

    if not isinstance(x, np.ndarray):
        raise TypeError("x should be a numpy array.")

    if len(x.shape) > 2:
        raise ValueError("x should be 1D or 2D.")

    header = {
        "FIELDS": [],
        "SIZE": [],
        "TYPE": [],
        "COUNT": [],
        "WIDTH": 0,
        "HEIGHT": 0,
        "POINTS": 0,
    }

    for descr in x.dtype.descr:

        dtype = np.dtype(descr[1])
        header["FIELDS"].append(descr[0].encode("UTF-8"))
        header["SIZE"].append(str(dtype.itemsize).encode("UTF-8"))
        header["TYPE"].append(dtype.kind.upper().encode("UTF-8"))

        cnt = 1
        if len(descr) == 3:
            cnt = np.prod(descr[2])

        header["COUNT"].append(str(cnt).encode("UTF-8"))

    if len(x.shape) == 1:
        header["WIDTH"] = x.shape[0]
        header["HEIGHT"] = 1
        header["POINTS"] = x.shape[0]
    else:
        header["WIDTH"] = x.shape[1]
        header["HEIGHT"] = x.shape[0]
        header["POINTS"] = x.shape[0] * x.shape[1]

    with open(f, "wb") as fd:
        _write_header(fd, header)
        fd.write(x.tobytes())
