import numpy as np

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


def is_camera_points_in_FOV(
    camera_pts,
    intrinsic,
    H,
    W,
    max_distance=None,
    min_distance=None,
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

    return mask


def points_to_depth_image(
    points,
    intrinsic,
    extrinsic,
    H,
    W,
    min_distance=None,
    max_distances=None,
    return_details=False,
):

    xyz = np.array(points)

    assert np.shape(intrinsic) == (3, 3)
    assert np.shape(extrinsic) == (4, 4)

    # apply extrinsic
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    xyz = R.T @ (xyz - t).T  # (3, N)

    mask = is_camera_points_in_FOV(
        xyz.T,
        intrinsic,
        H,
        W,
        max_distance=max_distances,
        min_distance=min_distance,
    )
    xyz = xyz[:, mask]  # (3, M)

    # apply intrinsic
    x, y, z = intrinsic @ xyz

    v = (x / z).astype(int)
    u = (y / z).astype(int)

    uv_indices = np.ravel_multi_index((u, v), (H, W))
    I = np.argsort(uv_indices)

    uv_indices = uv_indices[I]

    splits = np.nonzero(np.diff(uv_indices))[0] + 1
    splits = np.r_[0, splits, len(uv_indices)]

    D = utils_segmentation.segmented_min(
        z[I],
        s_ind=splits[:-1],
        e_ind=splits[1:],
    )

    u, v = np.unravel_index(uv_indices[splits[:-1]], (H, W))

    depth = np.ones((H, W)) * -1
    depth[u, v] = D

    if return_details:
        details = {
            "camera_pts": xyz.T,
            "mask": mask,
        }
        return depth, details

    return depth


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

    # equivalent to: extrinsic @ np.vstack([xyz, np.ones(H * W)])
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    xyz = R @ xyz + t[:, None]  # (3, H * W)

    return xyz.T  # (H * W, 3)
