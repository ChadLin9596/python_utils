"""
Segmentation related utility functions.

The functions in this module generally assume that `s_ind` and
`e_ind` are 1D arrays of the same length, and that for each i,
`s_ind[i] <= e_ind[i]`. The segmented operations can be treated as
numpy.ufunc.reduceat with custom started and ended indices. See:
(https://numpy.org/doc/1.26/reference/generated/numpy.ufunc.reduceat.html)

Dependency:
    - numpy_reduceat_ext
    - numpy
"""

import numpy as np
from numpy_reduceat_ext import numpy_argmax_reduceat, numpy_argmin_reduceat


def _numpy_argmax_reduceat(array, indices):
    """
    a slower vectorizing version (for reference and record only)
    modified from https://stackoverflow.com/questions/41833740
    """

    array = np.asarray(array)
    indices = np.asarray(indices)

    if np.any(np.diff(indices) < 0):
        msg = "`indices` must be monotonically increasing."
        raise ValueError(msg)

    max_value = np.max(array) + 1

    # create a shifted array enforcing further argsort operation will
    # not mixed indices across each segment.
    id_array = np.zeros(len(array), dtype=np.int64)
    id_array[indices] = 1
    id_array = np.cumsum(id_array)
    shift = max_value * id_array

    # the case when i < j, where i and j are adjacent index at indices
    I = np.argsort(array + shift)
    I_end_grp = np.r_[indices[1:], len(array)] - 1
    global_indices = I[I_end_grp]

    # the case when i == j, where i and j are adjacent index at indices
    splits = np.r_[indices, len(array)]
    J = np.where(splits[:-1] == splits[1:])[0]
    global_indices[J] = indices[J]

    return global_indices


def _numpy_argmin_reduceat(array, indices):
    """
    a slower vectorizing version (for reference and record only)
    modified from https://stackoverflow.com/questions/41833740
    """

    array = np.asarray(array)
    indices = np.asarray(indices)

    if np.any(np.diff(indices) < 0):
        msg = "`indices` must be monotonically increasing."
        raise ValueError(msg)

    max_value = np.max(array) + 1

    # create a shifted array enforcing further argsort operation will
    # not mixed indices across each segment.
    id_array = np.zeros(len(array), dtype=np.int64)
    id_array[indices] = 1
    id_array = np.cumsum(id_array)
    shift = max_value * id_array

    # the case when i < j, where i and j are adjacent index at indices
    I = np.argsort(array + shift, kind="stable")
    I_start_grp = indices
    global_indices = I[I_start_grp]

    # the case when i == j, where i and j are adjacent index at indices
    splits = np.r_[indices, len(array)]
    J = np.where(splits[:-1] == splits[1:])[0]
    global_indices[J] = indices[J]

    return global_indices


def segmented_sum(x, s_ind, e_ind):
    """
    Computes the summation along segments x[s_ind:e_ind, ...]

    Parameters
    ----------
    x : a numpy ndarray
    s_ind : a 1d array (n,)
    e_ind : a 1d array (n,)

    Return
    ------
    a numpy ndarray
    """
    x = np.array(x)
    s_ind = np.array(s_ind)
    e_ind = np.array(e_ind)
    max_num = x.shape[0]

    if s_ind.shape != e_ind.shape or len(s_ind.shape) != 1:
        raise ValueError("s_end and e_end must be 1d array.")

    if np.any(s_ind >= max_num) or np.any(e_ind > max_num):
        raise ValueError(
            "\n"
            "s_end must be [ 0, x.shape[axis] )\n"
            "e_end must be [ 0, x.shape[axis] ]"
        )

    if np.any((e_ind - s_ind) < 0):
        raise ValueError("s_ind cannot greater than e_ind")

    # numpy ufunc.reduceat limitation:
    # e_ind cannot equal greater to len(x)
    # the last elements will be added in a few line later
    I = np.nonzero(e_ind == max_num)[0]
    e_ind[I] = max_num - 1

    splits = np.vstack([s_ind, e_ind]).T.flatten()
    r = np.add.reduceat(x, splits, axis=0)[::2]

    # add the last elements back
    J = s_ind[I] != max_num - 1
    r[I[J]] += x.take(-1, axis=0)

    return r


def segmented_count(x, s_ind, e_ind):
    """
    Counts the number of elements along segments x[s_ind:e_ind].

    Parameters
    ----------
    x : np.ndarray
        The input array. (for checking max length of s_ind and e_ind)
    s_ind : np.ndarray of int
        The array of start indices for each window.
    e_ind : np.ndarray of int
        The array of end indices for each window (exclusive).

    Return
    ------
    a numpy ndarray

    Examples
    --------
    >>> x = [1, 2, 4, 8]
    >>> s_ind = [0, 1, 2]
    >>> e_ind = [3, 4, 4]
    >>> segmented_count(x, s_ind, e_ind)
    array([3, 3, 2])

    >>> x = [1, 2, 4, 8]
    >>> s_ind = [0, 1, 2]
    >>> e_ind = [1, 1, 4]
    >>> segmented_count(x, s_ind, e_ind)
    array([1, 1, 2])
    """
    x = np.array(x)
    s_ind = np.array(s_ind)
    e_ind = np.array(e_ind)
    max_num = x.shape[0]

    if s_ind.shape != e_ind.shape or len(s_ind.shape) != 1:
        raise ValueError("s_end and e_end must be 1d array.")

    if np.any(s_ind >= max_num) or np.any(e_ind > max_num):
        raise ValueError(
            "\n"
            "s_end must be [ 0, x.shape[axis] )\n"
            "e_end must be [ 0, x.shape[axis] ]"
        )

    if np.any((e_ind - s_ind) < 0):
        raise ValueError("s_ind cannot greater than e_ind")

    I = s_ind == e_ind
    e_ind[I] = e_ind[I] + 1
    return e_ind - s_ind


def segmented_mean(x, s_ind, e_ind):
    """
    Computes the average along segments x[s_ind:e_ind, ...]

    Parameters
    ----------
    x : a numpy ndarray
    s_ind : a 1d array (n,)
    e_ind : a 1d array (n,)

    Return
    ------
    a numpy ndarray
    """
    r = segmented_sum(x, s_ind, e_ind)
    c = segmented_count(x, s_ind, e_ind)

    assert len(r) == len(c)

    shp = [1] * len(r.shape)
    shp[0] = len(c)
    return 1.0 * r / c.reshape(shp)


def segmented_where(x, s_ind, e_ind, values):
    """
    Computes the indices where x[s_ind:e_ind] == values within each segment.
    """
    x = np.asarray(x)
    s_ind = np.asarray(s_ind)
    e_ind = np.asarray(e_ind)
    values = np.asarray(values)

    if s_ind.shape != e_ind.shape or s_ind.ndim != 1:
        raise ValueError(
            "s_ind and e_ind must be 1D arrays of the same shape."
        )

    results = []

    for s, e, val in zip(s_ind, e_ind, values):
        result = s + np.where(x[s:e] == val)[0]
        results.append(result)

    return results


def segmented_argmax(x, s_ind, e_ind):
    """
    Computes the argmax along segments x[s_ind:e_ind]

    Parameters
    ----------
    x : np.ndarray
    s_ind : np.ndarray of int
        The array of start indices for each window.
    e_ind : np.ndarray of int
        The array of end indices for each window (exclusive).

    Return
    ------
    np.ndarray of int pointing global indices of x
    """

    x = np.array(x)
    s_ind = np.array(s_ind)
    e_ind = np.array(e_ind)
    max_num = x.shape[0]

    if s_ind.shape != e_ind.shape or len(s_ind.shape) != 1:
        raise ValueError("s_end and e_end must be 1d array.")

    if np.any(s_ind >= max_num) or np.any(e_ind > max_num):
        raise ValueError(
            "\n"
            "s_end must be [ 0, x.shape[axis] )\n"
            "e_end must be [ 0, x.shape[axis] ]"
        )

    if np.any((e_ind - s_ind) < 0):
        raise ValueError("s_ind cannot greater than e_ind")

    x = np.concatenate([x, x[-1:]], axis=0)
    splits = np.vstack([s_ind, e_ind]).T.flatten()

    results = numpy_argmax_reduceat(x, splits)[::2]
    return results


def segmented_max(x, s_ind, e_ind, return_indices=False):
    """
    Computes the maximum along segments x[s_ind:e_ind, ...]

    Parameters
    ----------
    x : np.ndarray
    s_ind : np.ndarray of int
        The array of start indices for each window.
    e_ind : np.ndarray of int
        The array of end indices for each window (exclusive).
    return_indices : bool, optional
        Whether to return the correspoinding indices in the original
        array.

    Return
    ------
    np.ndarray
    (optional) indices of these max values in the original array
    """
    indices = segmented_argmax(x, s_ind, e_ind)

    if return_indices:
        return x[indices], indices
    return x[indices]


def segmented_argmin(x, s_ind, e_ind):
    """
    Computes the argmin along segments x[s_ind:e_ind, ...]

    Parameters
    ----------
    x : np.ndarray
    s_ind : np.ndarray of int
        The array of start indices for each window.
    e_ind : np.ndarray of int
        The array of end indices for each window (exclusive).

    Return
    ------
    np.ndarray of int pointing global indices of x
    """

    x = np.array(x)
    s_ind = np.array(s_ind)
    e_ind = np.array(e_ind)
    max_num = x.shape[0]

    if s_ind.shape != e_ind.shape or len(s_ind.shape) != 1:
        raise ValueError("s_end and e_end must be 1d array.")

    if np.any(s_ind >= max_num) or np.any(e_ind > max_num):
        raise ValueError(
            "\n"
            "s_end must be [ 0, x.shape[axis] )\n"
            "e_end must be [ 0, x.shape[axis] ]"
        )

    if np.any((e_ind - s_ind) < 0):
        raise ValueError("s_ind cannot greater than e_ind")

    x = np.concatenate([x, x[-1:]], axis=0)
    splits = np.vstack([s_ind, e_ind]).T.flatten()

    results = numpy_argmin_reduceat(x, splits)[::2]
    return results


def segmented_min(x, s_ind, e_ind, return_indices=False):
    """
    Computes the minimum along segments x[s_ind:e_ind, ...]
    Returns:
        - min values within each segment
        - (optional) indices of these min values in the original array
    """
    indices = segmented_argmin(x, s_ind, e_ind)

    if return_indices:
        return x[indices], indices
    return x[indices]


def compute_sliding_window_indices(N, window_size, same_size=True):
    """
    Compute start and end indices for sliding windows over
    a sequence of length `N`.

    Parameters
    ----------
    N : int
        The length of the sequence over which sliding windows are computed.
    window_size : int
        The size of each sliding window.
    same_size : bool, optional
        * (True) only full-sized windows are returned.
        * (False) partial windows near the boundaries are included as well.

    Returns
    -------
    s_ind : np.ndarray of int
        The array of start indices for each window.
    e_ind : np.ndarray of int
        The array of end indices for each window (exclusive).

    Examples
    --------
    >>> s_ind, e_ind = compute_sliding_window_indices(5, 3, same_size=True)
    s_ind: [0, 1, 2]
    e_ind: [3, 4, 5]

    >>> s_ind, e_ind = compute_sliding_window_indices(5, 4, same_size=True)
    s_ind: [0, 1]
    e_ind: [4, 5]

    >>> s_ind, e_ind = compute_sliding_window_indices(5, 6, same_size=True)
    s_ind: []
    e_ind: []

    >>> s_ind, e_ind = compute_sliding_window_indices(5, 3, same_size=False)
    s_ind:           [0, 0, 1, 2, 3]
    e_ind:           [2, 3, 4, 5, 5]
    -> segment_size: [2, 3, 3, 3, 2]

    >>> s_ind, e_ind = compute_sliding_window_indices(5, 4, same_size=False)
    s_ind:           [0, 0, 0, 1, 2, 3]
    e_ind:           [2, 3, 4, 5, 5, 5]
    -> segment_size: [2, 3, 4, 4, 3, 2]

    >>> s_ind, e_ind = compute_sliding_window_indices(5, 6, same_size=False)
    s_ind:           [0, 0, 0, 0, 1, 2]
    e_ind:           [3, 4, 5, 5, 5, 5]
    -> segment_size: [3, 4, 5, 5, 4, 3]
    """

    if window_size % 2:
        xs = np.arange(N)
    else:
        xs = np.arange(N + 1)

    s_ind = np.clip(xs - window_size // 2, 0, N)
    e_ind = np.clip(xs + window_size // 2 + window_size % 2, 0, N)

    if not same_size:
        return s_ind, e_ind

    length = e_ind - s_ind
    M = length == window_size

    return s_ind[M], e_ind[M]


def compute_sliding_window_indices_with_overlap(
    N,
    window_size,
    overlap_ratio=0.33,
):

    s_ind, e_ind = compute_sliding_window_indices(
        N,
        window_size,
        same_size=True,
    )

    overlap_size = round(window_size * overlap_ratio)
    stride = window_size - overlap_size

    I = np.arange(len(s_ind))
    I = np.r_[I[::stride], I[-1]]
    I = np.unique(I)

    s_ind = s_ind[I]
    e_ind = e_ind[I]

    return s_ind, e_ind


def sliding_window(x, window_size=3, same_size=False, method="mean"):
    """
    Apply sliding window operation on 1D array `x`. `window_size` &
    `same_size` control the sliding window indices, see
    `compute_sliding_window_indices` for details. `method` specifies
    the aggregation method within each window.

    Parameters
    ----------
    x : np.ndarray of shape (N,)
        The input 1D array.

    window_size : int, optional

    same_size : bool, optional

    method : str, optional
        - "argmin": index of minimum
        - "argmax": index of maximum
        - "max": maximum
        - "min": minimum
        - "count": count of elements
        - "sum": summation
        - "mean": average
    """

    func_map = {
        "sum": segmented_sum,
        "count": segmented_count,
        "mean": segmented_mean,
        "max": segmented_max,
        "min": segmented_min,
        "argmax": segmented_argmax,
        "argmin": segmented_argmin,
    }

    if method not in func_map:
        raise ValueError(f"Unknown method: {method}")

    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number")

    N = len(x)
    s_ind, e_ind = compute_sliding_window_indices(
        N,
        window_size,
        same_size=same_size,
    )

    func = func_map[method]
    results = func(x, s_ind, e_ind)
    assert len(results) == N

    return results
