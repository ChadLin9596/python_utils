import numpy as np

# 0 0000 | 1 0001 | 2 0010 | 3 0011
# 4 0100 | 5 0101 | 6 0110 | 7 0111
# 8 1000 | 9 1001 | A 1010 | B 1011
# C 1100 | D 1101 | E 1110 | F 1111


def _spread_bits(v):
    """Insert "00" between adjacent bits of the low 21 bits."""

    # assume v is encoded by uint64 and dim is 3,
    # making effective 21 bits (64 // 3)
    v = v.astype(np.uint64)

    # preserve info at the last 21 (1FFFFF) bits, thrown away others
    # bit: | 63 ... 21 | 20 ... 00 |
    # val: |     zeros |         a |
    v = v & np.uint64(0x1FFFFF)

    ####################################################################

    # bit: | 63 ... 53 | 52 ... 32 | 31 ... 21 | 20 ... 00 |
    # val: |     zeros |         a |     zeros |         a |
    v = v | (v << np.uint64(32))

    # 0x1F00000000FFFF
    # bit: | 63 ... 53 | 52 ... 48 | 47 ... 16 | 15 ... 00 |
    # val: |     zeros |      ones |     zeros |      ones |
    # ->
    # a = a1 | a2
    # bit: | 63 ... 53 | 52 ... 48 | 47 ... 16 | 15 ... 00 |
    # val: |     zeros |        a1 |     zeros |        a2 |
    v = v & np.uint64(0x1F00000000FFFF)

    ####################################################################

    # v << np.uint64(16)
    # bit: | 63 ... 32 | 31 ... 16 | 15 ... 00 |
    # val: |     zeros |        a2 |     zeros |
    # ->
    # bit: | 63 ... 53 | 52 ... 48 | 47 ... 32 | 31 ... 16 | 15 ... 00 |
    # val: |     zeros |        a1 |     zeros |        a2 |        a2 |
    v = v | (v << np.uint64(16))

    # 0x1F0000FF0000FF
    # bit: | 52 ... 48 | 47 ... 32 | 31 ... 24 | 23 ... 08 | 07 ... 00 |
    # val: |      ones |     zeros |      ones |     zeros |      ones |
    # ->
    # a2 = a21 | a22
    # bit: | 52 ... 48 | 47 ... 32 | 31 ... 24 | 23 ... 08 | 07 ... 00 |
    # val: |        a1 |     zeros |       a21 |     zeros |       a22 |
    v = v & np.uint64(0x1F0000FF0000FF)

    ####################################################################

    # bit: | 60 ... 56 | 52 ... 48 | 39 ... 32 | 31 ... 24 | 15 ... 08 |
    # val: |        a1 |        a1 |       a21 |       a21 |       a22 |
    # bit: | 07 ... 00 |
    # val: |       a22 |
    v = v | (v << np.uint64(8))

    # 0x100F00F00F00F00F
    # 00F: 0000 0000 1111
    # bit: |        60 | 51 ... 48 | 39 ... 36 | 27 ... 24 | 15 ... 12 |
    # val: |      ones |      ones |      ones |      ones |      ones |
    # bit: | 03 ... 00 |
    # val: |      ones |
    # ->
    # bit: |        60 | 51 ... 48 | 39 ... 36 | 27 ... 24 | 15 ... 12 |
    # val: |       a11 |       a12 |      a211 |      a212 |      a221 |
    # bit: | 03 ... 00 |
    # val: |      a222 |
    v = v & np.uint64(0x100F00F00F00F00F)

    ####################################################################

    # bit: |        60 | 55 ... 52 | 51 ... 48 |
    # val: |       a11 |       a12 |       a12 |
    # bit: | 43 ... 40 | 39 ... 36 | 31 ... 28 | 27 ... 24 |
    # val: |      a211 |      a211 |      a212 |      a212 |
    # bit: | 19 ... 16 | 15 ... 12 | 07 ... 04 | 03 ... 00 |
    # val: |      a221 |      a221 |      a222 |      a222 |
    v = v | (v << np.uint64(4))

    # 0C3: 0000 1100 0011
    v = v & np.uint64(0x10C30C30C30C30C3)

    ####################################################################

    # 249: 0010 0100 1001
    v = v | (v << np.uint64(2))
    v = v & np.uint64(0x1249249249249249)
    return v


###########
# Z-ORDER #
###########


def z_order_curve(xyz, res):

    _xyz = xyz - np.min(xyz, axis=0)
    _xyz = np.floor(_xyz / res)
    _xyz = _xyz.astype(np.uint64)

    x, y, z = _xyz.T

    _key = _spread_bits(x)
    _key = _key | (_spread_bits(y) << np.uint64(1))
    _key = _key | (_spread_bits(z) << np.uint64(2))

    indices = np.argsort(_key, kind="stable")
    return indices


#################
# HILBERT-CURVE #
#################


def _hilbert_encode(grid_coords, depth=21):

    assert len(np.shape(grid_coords)) == 2
    _, D = grid_coords.shape

    if D * depth > 64:
        msg = f"{D} dims x {depth} bits does not fit in uint64"
        raise ValueError(msg)

    one = np.uint64(1)
    axes = [grid_coords[:, i].astype(np.uint64) for i in range(D)]

    top_bit = one << np.uint64(depth - 1)
    level_bit = top_bit

    while level_bit > one:

        lower_bits = level_bit - one
        for i in range(D):
            in_upper_half = (axes[i] & level_bit) != 0
            diff_bits = (axes[0] ^ axes[i]) & lower_bits

            axes[0] = np.where(
                in_upper_half,
                axes[0] ^ lower_bits,
                axes[0] ^ diff_bits,
            )

            axes[i] = np.where(
                in_upper_half,
                axes[i],
                axes[i] ^ diff_bits,
            )
        level_bit = level_bit >> one

    for i in range(1, D):
        axes[i] = axes[i] ^ axes[i - 1]

    parity_mask = np.zeros_like(axes[0])
    level_bit = top_bit
    while level_bit > one:

        mask = (axes[-1] & level_bit) != 0
        parity_mask = np.where(
            mask,
            parity_mask ^ (level_bit - one),
            parity_mask,
        )
        level_bit = level_bit >> one

    axes = [axis ^ parity_mask for axis in axes]

    key = _spread_bits(axes[0]) << np.uint64(D - 1)
    for i in range(1, D):
        key = key | (_spread_bits(axes[i]) << np.uint64(D - 1 - i))
    return key


def hilbert_curve(xyz, res):

    _xyz = xyz - np.min(xyz, axis=0)
    _xyz = np.floor(_xyz / res)
    _xyz = _xyz.astype(np.uint64)

    _depth = int(np.max(_xyz)).bit_length()
    _depth = max(_depth, 1)

    _key = _hilbert_encode(_xyz, _depth)

    indices = np.argsort(_key, kind="stable")
    return indices
