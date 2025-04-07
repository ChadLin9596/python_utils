import os
import struct

import numpy as np

# TODO: refactor

# Reference:
# https://paulbourke.net/dataformats/ply/

STRUCTURE_TYPE = {
    "char": "b",
    "uchar": "B",
    "short": "h",
    "ushort": "H",
    "int": "i",
    "uint": "I",
    "float": "f",
    "double": "d",
}


def _parse_header(fd, return_raw=False):
    # Read header
    _header = []
    while True:
        line = fd.readline().decode("utf-8").strip()
        _header.append(line)
        if line == "end_header":
            break

    if return_raw:
        return _header

    # format the header
    header = {}
    elements = []  # use list to preserve the order of elements
    current_element = None

    for h in _header:

        if h.startswith("ply") or h.startswith("end_header"):
            continue

        key, value = h.split(maxsplit=1)

        if key == "format":
            format_type, version = value.split()
            header["format"] = format_type
            header["version"] = version

        elif key == "comment":
            prefix = header.get("comment", "")
            prefix += "\n" if len(prefix) > 0 else ""
            header["comment"] = prefix + value

        elif key == "element":
            if current_element is not None:
                elements.append(current_element)

            # initialize a new element
            name, count = value.split()
            current_element = {"name": name, "count": count, "properties": []}

        elif key == "property":
            assert current_element is not None

            if value.startswith("list"):
                _, list_type, index_type, name = value.split()
                current_element["properties"].append(
                    {
                        "type": "list",
                        "list_type": list_type,
                        "index_type": index_type,
                        "name": name,
                    }
                )
            else:
                prop_type, name = value.split()
                current_element["properties"].append(
                    {
                        "type": "property",
                        "prop_type": prop_type,
                        "name": name,
                    }
                )
        else:
            raise ValueError(f"Unknown header line: {h}")

    if current_element is not None:
        elements.append(current_element)
        header["elements"] = elements

    return header


def parse_header_by_file(f, return_size=False):
    if not isinstance(f, str):
        raise TypeError("f should be a string.")

    if not os.path.exists(f):
        raise OSError("%s not found" % f)

    with open(f, "rb") as fd:
        header = _parse_header(fd, return_raw=False)
        size = fd.tell()

    if return_size:
        return header, size
    return header


def _format_to_numpy_type(properties, format="ascii"):

    M = {
        "char": "i1",
        "uchar": "u1",
        "short": "i2",
        "ushort": "u2",
        "int": "i4",
        "uint": "u4",
        "float": "f4",
        "double": "f8",
    }

    assert np.iterable(properties)

    M_endianness = {
        "binary_little_endian": "<",
        "binary_big_endian": ">",
        "ascii": "",
    }

    dtype = []
    for prop in properties:

        endianness = M_endianness[format]

        name = prop["name"]
        prop_type = endianness + M[prop["prop_type"]]
        dtype.append((name, prop_type))

    return dtype


def expand_list_by_triangle_fan(indices):

    assert len(indices) >= 3

    triangles = []
    for i, j in zip(indices[1:], indices[2:]):
        triangles.append([indices[0], i, j])

    return triangles


def _read_ascii(filename):

    header, size = parse_header_by_file(filename, return_size=True)

    assert header["format"] == "ascii", "Only ascii format is supported."

    results = {}

    with open(filename, "r") as fd:
        fd.seek(size)

        for element in header["elements"]:

            name = element["name"]
            count = int(element["count"])
            properties = element["properties"]

            if element["properties"][0]["type"] == "property":

                dtype = _format_to_numpy_type(properties, format="ascii")
                array = np.ndarray(count, dtype=np.dtype(dtype))

                data = []
                for _ in range(count):
                    line = fd.readline().strip()
                    data.append(line.split())
                data = np.array(data)

                for (field, np_type), col in zip(dtype, data.T):
                    array[field] = col.astype(np_type)

                results[name] = array

            elif element["properties"][0]["type"] == "list":

                assert len(element["properties"]) == 1

                data_list = []
                for _ in range(count):
                    d = fd.readline().strip().split()
                    d = d[1:]
                    d = list(map(int, d))
                    data_list.extend(expand_list_by_triangle_fan(d))

                results[name] = data_list

    return results


def _read_binary(filename):

    header, size = parse_header_by_file(filename, return_size=True)

    results = {}
    with open(filename, "rb") as fd:
        fd.seek(size)

        is_little_endian = header["format"] == "binary_little_endian"
        byte_order = "<" if is_little_endian else ">"

        for element in header["elements"]:
            name = element["name"]
            count = int(element["count"])
            properties = element["properties"]

            if properties[0]["type"] == "property":

                dtype = _format_to_numpy_type(
                    properties,
                    format=header["format"],
                )
                dtype = np.dtype(dtype)

                data = fd.read(dtype.itemsize * count)
                array = np.frombuffer(data, dtype=dtype)

                if not is_little_endian:
                    array = array.byteswap().newbyteorder()

                results[name] = array

            elif properties[0]["type"] == "list":

                assert len(properties) == 1
                prop = properties[0]

                count_fmt = STRUCTURE_TYPE[prop["list_type"]]
                index_fmt_base = STRUCTURE_TYPE[prop["index_type"]]

                count_size = struct.calcsize(count_fmt)
                index_size = struct.calcsize(index_fmt_base)

                data_list = []
                for _ in range(count):

                    data = fd.read(count_size)
                    list_len = struct.unpack(byte_order + count_fmt, data)[0]

                    data = fd.read(list_len * index_size)
                    index_fmt = f"{byte_order}{list_len}{index_fmt_base}"
                    indices = struct.unpack(index_fmt, data)

                    data_list.extend(expand_list_by_triangle_fan(indices))

                results[name] = data_list

    return results


def read(filename, return_header=False):

    header = parse_header_by_file(filename)

    if header["format"] == "ascii":
        result = _read_ascii(filename)

    else:
        result = _read_binary(filename)

    if return_header:
        return result, header
    return result


def write(
    file,
    vertex,
    face=None,
    edge=None,
    comment="",
):
    """
    face should be a list of list of int
    """

    M = {
        "i1": "char",
        "u1": "uchar",
        "i2": "short",
        "u2": "ushort",
        "i4": "int",
        "u4": "uint",
        "f4": "float",
        "f8": "double",
    }

    with open(file, "wb") as fd:

        # Write header
        fd.write(b"ply\n")
        fd.write(b"format binary_little_endian 1.0\n")
        if comment:
            for c in comment.split("\n"):
                msg = "comment " + c + "\n"
                fd.write(msg.encode())

        fd.write(b"element vertex %d\n" % len(vertex))
        for name, typ in vertex.dtype.descr:
            np_type = np.dtype(typ).str[1:]
            ply_type = M[np_type]
            fd.write(f"property {ply_type} {name}\n".encode())

        if face is not None:
            fd.write(b"element face %d\n" % len(face))
            fd.write(b"property list uchar int vertex_index\n")

        if edge is not None:
            fd.write(b"element edge %d\n" % len(edge))
            for name, typ in edge.dtype.descr:
                np_type = np.dtype(typ).str[1:]
                ply_type = M[np_type]
                fd.write(f"property {ply_type} {name}\n".encode())

        fd.write(b"end_header\n")

        # Write body
        fd.write(vertex.tobytes())

        if face is not None:
            for f_item in face:
                N = len(f_item)
                fd.write(struct.pack("B", N))
                fd.write(struct.pack(f"{N}i", *f_item))

        if edge is not None:
            fd.write(edge.tobytes())
