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


def umeyama_ransac_alignment(
    src,
    dst,
    with_scale=True,
    max_iterations=1000,
    inlier_threshold=0.05,
    min_samples=4,
    seed=None,
):
    """
    RANSAC-robust Umeyama alignment.

    Estimates s, R, t such that dst ≈ s * R * src + t, while rejecting outlier
    correspondences.  After RANSAC, the winning model is refined by re-running
    umeyama_alignment on all inliers.

    Args:
        src: (N, D) source points.
        dst: (N, D) destination points (same ordering / correspondence as src).
        with_scale: whether to estimate scale (passed through to umeyama_alignment).
        max_iterations: maximum RANSAC iterations.
        inlier_threshold: residual distance below which a correspondence is an inlier.
        min_samples: number of correspondences sampled per hypothesis (>= D).
        seed: optional RNG seed for reproducibility.

    Returns:
        scale (float), R (D×D ndarray), t (D ndarray), inlier_mask (N bool ndarray)
    """
    assert src.shape == dst.shape
    assert (
        src.shape[0] >= min_samples
    ), "Need at least min_samples correspondences."

    rng = np.random.default_rng(seed)
    n = src.shape[0]

    best_inliers = np.zeros(n, dtype=bool)
    best_scale, best_R, best_t = (
        1.0,
        np.eye(src.shape[1]),
        np.zeros(src.shape[1]),
    )

    for _ in range(max_iterations):
        idx = rng.choice(n, size=min_samples, replace=False)
        scale, R, t = umeyama_alignment(
            src[idx], dst[idx], with_scale=with_scale
        )

        residuals = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)
        inliers = residuals < inlier_threshold

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_scale, best_R, best_t = scale, R, t

    # Refine on all inliers if we have enough
    if best_inliers.sum() >= min_samples:
        best_scale, best_R, best_t = umeyama_alignment(
            src[best_inliers], dst[best_inliers], with_scale=with_scale
        )
        residuals = np.linalg.norm(
            best_scale * (src @ best_R.T) + best_t - dst, axis=1
        )
        best_inliers = residuals < inlier_threshold

    return best_scale, best_R, best_t, best_inliers
