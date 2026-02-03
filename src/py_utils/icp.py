import numpy as np
import pptk.kdtree

from . import utils
from . import utils_align


def _build_kd_tree(points):
    return pptk.kdtree._build(points)


def nearest_neighbors(src_points, kdtree_or_dst_points, dmax=np.inf):

    kdtree = kdtree_or_dst_points
    if isinstance(kdtree, np.ndarray):
        kdtree = _build_kd_tree(kdtree)

    indices = pptk.kdtree._query(kdtree, src_points, k=1, dmax=dmax)

    src_indices = []
    dst_indices = []
    for src_idx, dst_idx in enumerate(indices):

        if len(dst_idx) == 0:
            continue

        src_indices.append(src_idx)
        dst_indices.append(dst_idx[0])

    return np.array(src_indices), np.array(dst_indices)


def iterative_closest_point(
    src_points,
    dst_points,
    initial_pose=np.eye(4),
    # optimization parameters
    max_iterations=20,
    max_distance=np.inf,
    min_delta=1e-6,
    min_mean_distance=1e-6,
    verbose=False,
    return_details=False,
):

    assert src_points.shape[1] == dst_points.shape[1]
    assert len(src_points.shape) == 2

    def xprint(*args):
        if verbose:
            print(*args)

    num_dim = src_points.shape[1]

    R = initial_pose[:3, :3]
    t = initial_pose[:3, 3]
    src_points = src_points @ R.T + t

    dst_kd_tree = _build_kd_tree(dst_points)
    nn_kwargs = {"kdtree_or_dst_points": dst_kd_tree, "dmax": max_distance}

    distances = []
    src_closest_indices = []
    dst_closest_indices = []
    Rs = []
    ts = []

    prog = utils.ProgressTimer(verbose=verbose)
    prog.tic(max_iterations)
    for _ in range(max_iterations):

        src_indices, dst_indices = nearest_neighbors(src_points, **nn_kwargs)
        src_matched = src_points[src_indices]
        dst_matched = dst_points[dst_indices]

        if len(src_indices) < num_dim:
            msg = f"ICP stopped early: only {len(src_indices)} "
            msg += "correspondences found."
            xprint(msg)
            break

        # compute distance and check convergence
        d = np.sqrt(np.sum((src_matched - dst_matched) ** 2, axis=1))
        d = np.mean(d)

        if d < min_mean_distance:
            msg = "ICP converged: "
            msg += f"mean distance {d:.6f} below threshold."
            xprint(msg)
            break

        prev_d = distances[-1] if len(distances) > 0 else np.inf
        diff_d = np.abs(prev_d - d)
        if diff_d < min_delta:
            msg = "ICP converged: "
            msg += f"mean distance change {diff_d:.6f} below threshold."
            xprint(msg)
            break

        # update src_points
        scale, R_delta, t_delta = utils_align.umeyama_alignment(
            src_matched,
            dst_matched,
            with_scale=False,
        )

        src_points = (src_points @ R_delta.T) + t_delta

        # update overall pose
        R = R_delta @ R
        t = R_delta @ t + t_delta

        # store iteration details
        distances.append(d)
        src_closest_indices.append(src_indices)
        dst_closest_indices.append(dst_indices)
        Rs.append(R.copy())
        ts.append(t.copy())

        prog.toc()

    details = {
        "distances": distances,
        "src_closest_indices": src_closest_indices,
        "dst_closest_indices": dst_closest_indices,
        "Rs": Rs,
        "ts": ts,
    }

    if return_details:
        return R, t, details

    return R, t
