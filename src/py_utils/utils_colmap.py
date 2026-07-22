import numpy as np


def _parse_shp(f, return_pos=True):
    # assume it is encoded by Width&Height&Depth&

    shp = b""
    _n = 0
    _pos = 0

    with open(f, "rb") as fd:

        while _n < 3:
            c = fd.read(1)
            shp += c
            if c == b"&":
                _n += 1

        _pos = fd.tell()

    shp = shp.decode("ascii").split("&")[:3]
    shp = list(map(int, shp))

    if return_pos:
        return shp, _pos

    return shp


def read_bin(f):

    shp, _pos = _parse_shp(f, return_pos=True)
    with open(f, "rb") as fd:
        fd.seek(_pos)
        _data = fd.read()

    expected_size = np.prod(shp) * 4  # number of bytes
    assert len(_data) >= expected_size

    _data = _data[:expected_size]
    array = np.frombuffer(_data, dtype=np.float32)
    array = array.reshape(shp, order="F").transpose(1, 0, 2)
    return array
