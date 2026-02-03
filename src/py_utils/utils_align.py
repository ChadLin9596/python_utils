import numpy as np


def umeyama_alignment(src, dst, with_scale=True):
    """
    Perform Umeyama alignment to find s (scale), R (rotation), and
    t (translation) that align src to dst.
    dst = s * R * src + t
    """

    assert src.shape == dst.shape

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov_matrix = dst_centered.T @ src_centered / src.shape[0]
    U, D, Vt = np.linalg.svd(cov_matrix)

    R = U @ Vt
    # Ensure a proper rotation (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    if with_scale:
        var_src = np.var(src_centered, axis=0).sum()
        scale = np.sum(D) / var_src
    else:
        scale = 1.0

    t = mu_dst - scale * R @ mu_src

    return scale, R, t
