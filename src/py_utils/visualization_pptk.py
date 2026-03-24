import pptk
import scipy.interpolate
import numpy as np


###########
# Checker #
###########


def _check_rgb(rgb, msg="RGB values must be in the range [0, 1]"):

    if np.any(rgb < 0) or np.any(rgb > 1) or np.shape(rgb)[-1] != 3:
        raise ValueError(msg)


def _check_alpha(alpha, msg="Alpha values must be in the range [0, 1]"):

    if np.max(alpha) > 1 or np.min(alpha) < 0:
        raise ValueError(msg)


########################################
# Supplementary Points/Color Generator #
########################################


def make_coordinates(xyz, R, length=1.0, number=20):
    # TODO: array shape check consistence

    steps = np.linspace(0, length, number)  # (number, )

    R = R.transpose([0, 2, 1])

    N = len(xyz)

    # (N, 1, 1, 3) + (N, 3, 1, 3) * (number, 1) = (N, 3, number, 3)
    points = xyz[..., None, None, :] + R[..., None, :] * steps[:, None]

    colors = np.array(
        [
            [1.0, 0.0, 0.0],  # red
            [0.0, 1.0, 0.0],  # green
            [0.0, 0.0, 1.0],  # blue
        ]
    )

    # (N, 3, number, 1) * (1, 3, 1, 3)
    colors = np.ones((N, 3, number, 1)) * colors[:, None, :]

    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    return points, colors


def make_lines(src, dst, start=True, end=True, eps=0.005, return_splits=False):
    """
    TODO add docstring and comments
    """

    V = dst - src
    L = np.sqrt(np.sum(V**2, axis=-1))

    num = np.maximum(2, np.floor(L / eps).astype(np.int64)) + 1
    splits = np.r_[0, np.cumsum(num)]

    xyz = np.empty((splits[-1], 3), dtype=np.float64)
    for i, j, n, s in zip(src, dst, num, splits[:-1]):
        xyz[s : s + n] = np.linspace(i, j, n)

    I = np.ones(splits[-1], dtype=np.bool_)

    if not start:
        I[splits[:-1]] = 0
        num = num - 1

    if not end:
        I[splits[1:] - 1] = 0
        num = num - 1

    if return_splits:
        return xyz[I], np.r_[0, np.cumsum(num)]
    return xyz[I]


def make_polygon(xyz, eps=0.005):
    A = xyz
    B = np.vstack([xyz[1:], xyz[:1]])
    L = make_lines(A, B, start=True, end=False, eps=eps)
    return L


def make_polyline(xyz, eps=0.005):
    L = make_lines(xyz[:-1], xyz[1:], start=True, end=False, eps=eps)
    return np.vstack([L, xyz[-1:]])


def _make_bounding_box_vertices(lx, ly, lz):
    """
    Notes:
    ------
        4----0
       /|   /|
      / 5--/-1
     / /  / /
    7----3 /  z y
    |/   |/   |/
    6----2    .--x
    """

    vertices = np.empty((8, 3), dtype=np.float64)

    vertices[0:4, 0] = lx
    vertices[4:8, 0] = -lx
    vertices[[0, 1, 4, 5], 1] = ly
    vertices[[2, 3, 6, 7], 1] = -ly
    vertices[[0, 3, 4, 7], 2] = lz
    vertices[[1, 2, 5, 6], 2] = -lz

    vertices = vertices / 2.0
    return vertices


def make_bounding_box_vertices(lxs, lys, lzs):
    """
    Notes:
    ------
    ```
        4----0
       /|   /|
      / 5--/-1
     / /  / /
    7----3 /  z y
    |/   |/   |/
    6----2    .--x
    ```
    """

    if np.isscalar(lxs) and np.isscalar(lys) and np.isscalar(lzs):
        return _make_bounding_box_vertices(lxs, lys, lzs)

    shp = np.shape(lxs)
    vertices = np.empty(shp + (8, 3), dtype=np.float64)

    vertices[..., 0:4, 0] = lxs[:, None]
    vertices[..., 4:8, 0] = -lxs[:, None]
    vertices[..., [0, 1, 4, 5], 1] = lys[:, None]
    vertices[..., [2, 3, 6, 7], 1] = -lys[:, None]
    vertices[..., [0, 3, 4, 7], 2] = lzs[:, None]
    vertices[..., [1, 2, 5, 6], 2] = -lzs[:, None]

    vertices = vertices / 2.0
    return vertices


def _make_bounding_box_lines(vertices, eps=0.005):
    """
    xyz: (N, 3)
    lx:  (N,)
    ly:  (N,)
    lz:  (N,)
    """

    assert np.shape(vertices) == (8, 3)

    I = [0, 1, 5, 4]
    J = [3, 2, 6, 7]
    points = np.vstack(
        [
            make_polygon(vertices[I], eps=eps),
            make_polygon(vertices[J], eps=eps),
            make_lines(
                vertices[I], vertices[J], start=False, end=False, eps=eps
            ),
        ]
    )
    return points


def make_bounding_boxes_lines(vertices, eps=0.005):

    assert np.shape(vertices)[-2:] == (8, 3)

    if len(np.shape(vertices)) == 2:
        points = _make_bounding_box_lines(vertices, eps=eps)
        return [points]

    points = []
    for vertex in vertices:
        points.append(_make_bounding_box_lines(vertex, eps=eps))
    return points


def make_color(s_min, s_max, color_map=None):
    """
    TODO: add docstring
    """

    # default color map(jet)
    _colormap = np.array(
        [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    )

    if color_map is None:
        color_map = _colormap

    if s_min == s_max:
        return lambda x: color_map[0]

    def wrap(x):
        foo = scipy.interpolate.interp1d(
            np.linspace(s_min, s_max, len(color_map)), color_map, axis=0
        )

        x = np.minimum(x, s_max)
        x = np.maximum(x, s_min)
        return foo(x)

    return wrap


###################
# Standardization #
###################


def _normalized_attributes(attributes, scale=[0, 100], color_map=None):

    f_make_color = make_color(*scale, color_map=color_map)

    rgbas = []
    for attr in attributes:

        attr = np.asarray(attr)

        # 1st case: if attr is a 2D array with 4 columns, treat it as RGBA
        if attr.ndim == 2 and attr.shape[1] == 4:

            _check_rgb(attr[:, :3])
            _check_alpha(attr[:, 3])
            rgbas.append(attr)
            continue

        # 2nd case: if attr is a 2D array with 3 columns, treat it as RGB
        if attr.ndim == 2 and attr.shape[1] == 3:

            _check_rgb(attr)
            alpha = np.ones(attr.shape[0])
            rgba = np.vstack([attr.T, alpha]).T
            rgbas.append(rgba)
            continue

        # 3rd case: if attr is a 2D array with 2 columns, treat is as [attr, A]
        if attr.ndim == 2 and attr.shape[1] == 2:

            rgb = f_make_color(attr[:, 0])
            alpha = attr[:, 1]

            msg = "color_map should be an (n, 3) array in the range [0, 1]"
            _check_rgb(rgb, msg)
            _check_alpha(alpha)

            rgba = np.vstack([rgb.T, alpha]).T
            rgbas.append(rgba)
            continue

        # 4th case: if attr is a 2D array with 1 column or 1D array
        #           treat it as attr
        if (attr.ndim == 2 and attr.shape[1] == 1) or attr.ndim == 1:
            attr = attr.flatten()
            rgb = f_make_color(attr)
            msg = "color_map should be an (n, 3) array in the range [0, 1]"
            _check_rgb(rgb, msg)

            alpha = np.ones(attr.shape[0])
            rgba = np.vstack([rgb.T, alpha]).T
            rgbas.append(rgba)
            continue

        raise ValueError(
            "Attributes must be a 2D array with 1, 2, 3, or 4 columns, "
            "or a 1D array. Received shape: {}".format(np.shape(attr))
        )

    return np.vstack(rgbas)


def _process_pcds(*pcds, scale=[0, 100], color_map=None):
    """
    Process point cloud data to extract xyz and attributes.
    Returns a tuple of (xyz, attributes).
    """

    if scale is not None and len(scale) != 2:
        raise ValueError("scale must be a tuple of two floats")

    xyzs = []
    attributes = []

    for p in pcds:
        assert len(p) == 2
        assert len(p[0]) == len(p[1])
        xyzs.append(p[0])
        attributes.append(p[1])

    splits = np.r_[0, np.cumsum([len(x) for x in xyzs])]

    xyz = np.vstack(xyzs)
    rgba = _normalized_attributes(attributes, scale=scale, color_map=color_map)

    rgbas = []
    for i, j in zip(splits[:-1], splits[1:]):

        mask_rgba = rgba.copy()
        mask_rgba[:, 3] = 0  # Set alpha to 0 for the entire array
        mask_rgba[i:j, 3] = rgba[i:j, 3]
        rgbas.append(mask_rgba)

    return xyz, rgbas


##############
# Visualizer #
##############


def plot_multiple_pcds(*pcds, scale=[0, 100], color_map=None):
    """
    TODO: add docstring
    """

    xyz, rgbas = _process_pcds(*pcds, scale=scale, color_map=color_map)
    v = pptk.viewer(xyz, debug=False)
    v.attributes(*rgbas)

    return v


def plot_matching_result(
    src_before,
    src_after,
    dst,
    matches=None,
    src_color=np.r_[1.0, 1.0, 0.2],
    dst_color=np.r_[0.2, 0.2, 1.0],
    line_color=np.r_[0.2, 1.0, 0.2],
    line_eps=0.005,
):

    if src_color.ndim == 1:
        src_color = np.tile(src_color[None, :], (len(src_before), 1))
        # src_color_after = np.tile(src_color[None, :], (len(src_after), 1))

    if dst_color.ndim == 1:
        dst_color = np.tile(dst_color[None, :], (len(dst), 1))

    # assert src_color_before.shape == (len(src_before), 3)
    # assert src_color_after.shape == (len(src_after), 3)
    # assert dst_color.shape == (len(dst), 3)

    before_xyzs = [src_before, dst]
    before_colors = [src_color, dst_color]

    after_xyzs = [src_after, dst]
    after_colors = [src_color, dst_color]

    # prepare matching lines
    if matches is not None:
        src_before_matched = src_before[matches[:, 0]]
        src_after_matched = src_after[matches[:, 0]]
        dst_matched = dst[matches[:, 1]]

        line_before_xyz = make_lines(
            src_before_matched,
            dst_matched,
            start=False,
            end=False,
            eps=line_eps,
        )

        line_after_xyz = make_lines(
            src_after_matched,
            dst_matched,
            start=False,
            end=False,
            eps=line_eps,
        )

        line_before_color = np.tile(line_color, (len(line_before_xyz), 1))
        line_after_color = np.tile(line_color, (len(line_after_xyz), 1))

        before_xyzs.append(line_before_xyz)
        before_colors.append(line_before_color)

        after_xyzs.append(line_after_xyz)
        after_colors.append(line_after_color)

    v = plot_multiple_pcds(
        (np.vstack(before_xyzs), np.vstack(before_colors)),
        (np.vstack(after_xyzs), np.vstack(after_colors)),
    )
    return v
