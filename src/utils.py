import numpy as np

def Q_to_R(q):
    if np.shape(q)[-1] != 4:
        raise ValueError('shape of q must be (..., 4)')

    # Normalize quaternions
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.empty(np.shape(q)[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2*y**2 - 2*z**2
    R[..., 0, 1] = 2*x*y - 2*w*z
    R[..., 0, 2] = 2*x*z + 2*w*y
    R[..., 1, 0] = 2*x*y + 2*w*z
    R[..., 1, 1] = 1 - 2*x**2 - 2*z**2
    R[..., 1, 2] = 2*y*z - 2*w*x
    R[..., 2, 0] = 2*x*z - 2*w*y
    R[..., 2, 1] = 2*y*z + 2*w*x
    R[..., 2, 2] = 1 - 2*x**2 - 2*y**2
    return R
