import cv2
import numpy as np
import scipy.ndimage
import scipy.spatial
import skimage.draw

from . import utils
from . import utils_segmentation


def clamp_image(img):
    return np.minimum(np.maximum(img, 0), 1.0)


def overlay_image(image, layer, ratio=0.5, mask=None):

    if mask is None:
        output = utils.merge_two_array(image, layer, ratio=ratio)
        return clamp_image(output)

    mask = np.array(mask).astype(bool)

    if image.shape[: len(mask.shape)] != mask.shape:
        raise ValueError(
            f"shape of image {image.shape} and "
            f"mask {mask.shape} is not compatible"
        )

    output = image.copy()
    output[mask] = utils.merge_two_array(output, layer, ratio=ratio)[mask]
    return clamp_image(output)


def center_crop_image(image, target_shapes):
    """
    Center crops an image to the specified size.

    Parameters:
    - image: a NumPy array of shape (height, width, channels).
    - target_shapes: (new_height, new_width) after cropping.

    Returns:
    - a NumPy array of shape (new_height, new_width, channels)
    """
    image = np.array(image)

    origin_shape = image.shape
    if len(origin_shape) == 2:
        image = image[..., None]

    if len(image.shape) != 3:
        raise ValueError

    height, width, _ = image.shape
    new_height, new_width = target_shapes

    # Calculate margins to remove from each side
    start_h = height // 2 - new_height // 2
    start_w = width // 2 - new_width // 2

    end_h = start_h + new_height
    end_w = start_w + new_width

    image = image[start_h:end_h, start_w:end_w, :]
    if len(origin_shape) == 2:
        image = image[..., 0]

    return image


def translate_image(img, tx, ty):
    """
    # modified from
    # https://stackoverflow.com/questions/63367506/image-translation-using-numpy
    """

    M, N = img.shape[:2]

    tx = max(min(tx, M), -M)
    ty = max(min(ty, N), -N)

    src_row_min = max(-tx, 0)
    src_row_max = M - max(tx, 0)
    src_col_min = max(-ty, 0)
    src_col_max = N - max(ty, 0)

    dst_row_min = max(tx, 0)
    dst_row_max = M + min(tx, 0)
    dst_col_min = max(ty, 0)
    dst_col_max = N + min(ty, 0)

    result = np.zeros_like(img)
    foo = img[src_row_min:src_row_max, src_col_min:src_col_max]
    result[dst_row_min:dst_row_max, dst_col_min:dst_col_max] = foo

    return result


def _trans_from_world_to_camera(points, extrinsic):
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    points = R.T @ (points - t).T
    points = points.T
    return points


def _trans_from_camera_to_world(points, extrinsic):
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    points = R @ points.T + t[:, None]
    points = points.T
    return points


def _indices_to_each_pixel(camera_points, intrinsic, H, W):

    x, y, z = intrinsic @ camera_points.T
    v = (x / z).astype(int)
    u = (y / z).astype(int)

    uv_indices = np.ravel_multi_index((u, v), (H, W))
    sorted_indices = np.argsort(uv_indices)
    uv_indices = uv_indices[sorted_indices]

    splits = np.nonzero(np.diff(uv_indices))[0] + 1
    splits = np.r_[0, splits, len(uv_indices)]

    return sorted_indices, splits


def indices_to_closest_points(camera_points, intrinsic, H, W):

    if len(camera_points) == 0:
        return np.array([], dtype=int)

    sorted_indices, splits = _indices_to_each_pixel(
        camera_points, intrinsic, H, W
    )

    # get the depth and indices from the closest points of each pixel
    depth, min_indices = utils_segmentation.segmented_min(
        camera_points[sorted_indices][:, 2],
        s_ind=splits[:-1],
        e_ind=splits[1:],
        return_indices=True,
    )
    min_indices = np.r_[[i[0] for i in min_indices]]

    return sorted_indices[min_indices]


def is_points_in_FOV(
    points,
    intrinsic,
    extrinsic,
    H,
    W,
    max_distance=None,
    min_distance=None,
    closest_only=False,
):

    mask = is_camera_points_in_FOV(
        _trans_from_world_to_camera(points, extrinsic),
        intrinsic,
        H,
        W,
        max_distance=max_distance,
        min_distance=min_distance,
        closest_only=closest_only,
    )

    return mask


def is_camera_points_in_FOV(
    camera_pts,
    intrinsic,
    H,
    W,
    max_distance=None,
    min_distance=None,
    closest_only=False,
):

    img_pts = camera_pts / camera_pts[:, 2:]
    img_pts = img_pts @ intrinsic.T

    # fmt: off
    mask = (
        (img_pts[:, 0] >= 0) &
        (img_pts[:, 0] < W) &
        (img_pts[:, 1] >= 0) &
        (img_pts[:, 1] < H) &
        (camera_pts[:, 2] > 0)
    )
    # fmt: on

    if max_distance is not None:
        mask &= camera_pts[:, 2] < max_distance

    if min_distance is not None:
        mask &= camera_pts[:, 2] > min_distance

    if closest_only:
        indices = indices_to_closest_points(camera_pts[mask], intrinsic, H, W)
        new_mask = np.zeros_like(mask)
        new_mask[np.where(mask)[0][indices]] = True
        mask = new_mask

    return mask


def points_to_index_map(
    points,
    intrinsic,
    extrinsic,
    H,
    W,
    min_distance=None,
    max_distance=None,
):

    assert np.shape(intrinsic) == (3, 3)
    assert np.shape(extrinsic) == (4, 4)

    cam_xyz = _trans_from_world_to_camera(points, extrinsic)
    FOV_mask = is_camera_points_in_FOV(
        cam_xyz,
        intrinsic,
        H,
        W,
        max_distance=max_distance,
        min_distance=min_distance,
        closest_only=True,
    )

    v, u, z = intrinsic @ cam_xyz[FOV_mask].T
    u = (u / z).astype(int)
    v = (v / z).astype(int)

    index_map = np.full((H, W), -1, dtype=np.int32)
    index_map[u, v] = np.arange(len(points))[FOV_mask]

    return index_map


def points_to_point_map(
    points,
    intrinsic,
    extrinsic,
    H,
    W,
    invalid_value=-1,
    min_distance=None,
    max_distance=None,
    return_details=False,
    other_attrs=[],
):

    world_xyz = np.array(points)

    assert np.shape(intrinsic) == (3, 3)
    assert np.shape(extrinsic) == (4, 4)

    # filter out points outside of FOV
    FOV_mask = is_points_in_FOV(
        world_xyz,
        intrinsic,
        extrinsic,
        H,
        W,
        max_distance=max_distance,
        min_distance=min_distance,
        closest_only=True,
    )
    world_xyz = world_xyz[FOV_mask]

    # convert points to camera coordinate
    cam_xyz = _trans_from_world_to_camera(world_xyz, extrinsic)

    # fill the depth image
    v, u, z = intrinsic @ cam_xyz.T
    u = (u / z).astype(int)
    v = (v / z).astype(int)

    point_map = np.full((H, W, 3), invalid_value, dtype=np.float64)
    point_map[u, v] = world_xyz

    # form the details for further usage
    details = {
        "uv": (u, v),
        "FOV_mask": FOV_mask,
    }

    if len(other_attrs):

        u, v = details["uv"]
        FOV_mask = details["FOV_mask"]

        output_attrs = []
        for attrs in other_attrs:
            assert len(attrs) == len(points)
            attrs = np.array(attrs)[FOV_mask]

            attrs_img = np.full(
                (H, W, *attrs.shape[1:]),
                invalid_value,
                dtype=attrs.dtype,
            )

            attrs_img[u, v] = attrs
            output_attrs.append(attrs_img)

    output = (point_map,)
    if len(other_attrs):
        output = output + tuple(output_attrs)

    if return_details:
        return *output, details

    if len(output) == 1:
        return output[0]
    return output


def points_to_depth_image(
    points,
    intrinsic,
    extrinsic,
    H,
    W,
    invalid_value=-1,
    min_distance=None,
    max_distance=None,
    return_details=False,
    other_attrs=[],
):

    output = points_to_point_map(
        points,
        intrinsic,
        extrinsic,
        H,
        W,
        invalid_value=invalid_value,
        min_distance=min_distance,
        max_distance=max_distance,
        return_details=True,
        other_attrs=other_attrs,
    )

    point_map = output[0]  # (H, W, 3)
    details = output[-1]  # {"uv": (u, v), "FOV_mask": FOV_mask}

    # ((H, W, ?), ...) if len(other_attrs) > 0 else empty tuple
    output_attrs = output[1:-1]

    u, v = details["uv"]

    # convert points to camera coordinate
    world_points = point_map[u, v]
    world_points = world_points.reshape(-1, 3)
    cam_xyz = _trans_from_world_to_camera(world_points, extrinsic)

    depth_img = np.full((H, W), invalid_value, dtype=np.float64)
    depth_img[u, v] = cam_xyz[:, 2]

    output = (depth_img,)
    if len(other_attrs):
        output = output + output_attrs

    if return_details:
        return *output, details

    if len(output) == 1:
        return output[0]
    return output


def depth_image_to_points(
    depth,
    intrinsic,
    extrinsic,
    H,
    W,
    offset=0.5,
    scale=1.0,
):

    depth = np.array(depth)

    assert np.prod(depth.shape) == 1.0 * H * W
    assert np.shape(intrinsic) == (3, 3)
    assert np.shape(extrinsic) == (4, 4)

    # x = [
    #     [0, 1, 2, ..., W-1],
    #     [0, 1, 2, ..., W-1],
    #     ...,
    #     [0, 1, 2, ..., W-1],
    # ]

    # y = [
    #     [  0,   0,   0, ...,   0],
    #     [  1,   1,   1, ...,   1],
    #     ...,
    #     [H-1, H-1, H-1, ..., H-1],
    # ]

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.flatten() + offset
    y = y.flatten() + offset
    depth = depth.flatten() * scale

    xyz = np.vstack([x, y, np.ones_like(x)])  # (3, H * W)
    xyz = np.linalg.inv(intrinsic) @ xyz  # (3, H * W)
    xyz = xyz * depth  # (3, H * W)
    xyz = xyz.T  # (H * W, 3)

    xyz = _trans_from_camera_to_world(xyz, extrinsic)
    return xyz


def _inpaint_by_conv_interpolation(
    arr,
    kernel_size=3,
    missing_condition_or_mask=lambda x: x == -1,
):

    if callable(missing_condition_or_mask):
        mask = missing_condition_or_mask(arr)
    else:
        mask = missing_condition_or_mask

    assert np.shape(mask) == np.shape(arr)

    kernel = np.ones((kernel_size, kernel_size))

    valid_mask = (~mask).astype(np.float32)

    valid_arr = arr * valid_mask

    kwargs = {"mode": "constant", "cval": 0.0}
    valid_sum = scipy.ndimage.convolve(valid_arr, kernel, **kwargs)
    valid_cnt = scipy.ndimage.convolve(valid_mask, kernel, **kwargs)

    out_arr = arr.copy()
    out_arr[mask] = valid_sum[mask] / np.maximum(valid_cnt[mask], 1)

    mask = valid_cnt == 0
    out_arr[mask] = arr[mask]
    return out_arr


def inpaint_by_conv_interpolation(
    arr,
    kernel_size=3,
    missing_condition_or_mask=lambda x: x == -1,
):

    func = _inpaint_by_conv_interpolation
    kwargs = {
        "kernel_size": kernel_size,
        "missing_condition_or_mask": missing_condition_or_mask,
    }

    if arr.ndim == 2:
        return func(arr, **kwargs)

    out_arr = [func(arr[..., i], **kwargs) for i in range(arr.shape[-1])]
    out_arr = np.dstack(out_arr)
    return out_arr


def hsv_to_rgb(hsv):
    """
    it is equivalent to matplotlib.colors.hsv_to_rgb
    TOTHINK: why not use matplotlib.colors.hsv_to_rgb?
    """

    h, s, v = np.moveaxis(hsv, -1, 0)  # Split into components

    i = (h * 6).astype(int)  # Sector index
    f = (h * 6) - i  # Fractional part
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    i = i % 6

    conditions = [
        (i == 0, (v, t, p)),
        (i == 1, (q, v, p)),
        (i == 2, (p, v, t)),
        (i == 3, (p, q, v)),
        (i == 4, (t, p, v)),
        (i == 5, (v, p, q)),
    ]

    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)

    for condition, (rc, gc, bc) in conditions:
        mask = condition
        r[mask], g[mask], b[mask] = rc[mask], gc[mask], bc[mask]

    return np.stack([r, g, b], axis=-1)


def fill_sparse_boolean_by_convex_hull(sparse_mask):
    # Create blank mask
    mask = np.zeros_like(sparse_mask, dtype=np.uint8)

    # Extract (row, col) of True pixels
    points = np.argwhere(sparse_mask)

    if len(points) < 3:
        # Too few points for a hull — return empty mask
        return mask

    # Compute convex hull using OpenCV (expects (x, y) = (col, row))
    points_cv = points[:, ::-1].astype(np.int32)  # Convert to (x, y)
    hull = cv2.convexHull(points_cv)

    # Fill the convex hull
    cv2.fillPoly(mask, [hull], color=1)

    return mask


def fill_sparse_boolean_by_delaunay(sparse_mask, return_triangles=False):

    # Create blank mask
    mask = np.zeros_like(sparse_mask, dtype=np.uint8)

    # Extract (row, col) of True pixels
    points = np.argwhere(sparse_mask)

    if len(points) < 3:
        # Too few points for a hull — return empty mask
        return mask

    tri = scipy.spatial.Delaunay(points)

    for simplex in tri.simplices:
        triangle = points[simplex]
        rr, cc = skimage.draw.polygon(
            triangle[:, 0], triangle[:, 1], shape=mask.shape
        )
        mask[rr, cc] = True

    if return_triangles:
        return mask, tri.simplices
    return mask


def update_intrinsics_by_resized(K, orig_shape, resized_shape):

    # Compute scale factors for resizing
    scale_x = resized_shape[1] / orig_shape[1]  # W' / W
    scale_y = resized_shape[0] / orig_shape[0]  # H' / H

    # Scale the intrinsic matrix due to resizing
    K_resized = K.copy()
    K_resized[0, 0] *= scale_x  # fx'
    K_resized[1, 1] *= scale_y  # fy'
    K_resized[0, 2] *= scale_x  # cx'
    K_resized[1, 2] *= scale_y  # cy'

    return K_resized
