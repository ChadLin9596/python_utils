import numpy as np

import utils

def clamp_image(img):
    return np.minimum(np.maximum(img, 0), 1.)

def overlay_image(image, layer, ratio=0.5, mask=None):

    if mask is None:
        output = utils.merge_two_array(image, layer, ratio=ratio)
        return clamp_image(output)

    mask = np.array(mask).astype(bool)

    if image.shape[:len(mask.shape)] != mask.shape:
        raise ValueError(
            f'shape of image {image.shape} and '
            f'mask {mask.shape} is not compatible')

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
