import numpy as np
import warnings

try:
    from numpy_reduceat_ext import _argmax, _argmin

    numpy_argmax_reduceat = _argmax.argmax_reduceat
    numpy_argmin_reduceat = _argmin.argmin_reduceat

except:
    warnings.warn(
        (
            "numpy_reduceat_ext C extension not found; "
            "using slower pure-Python fallback. (~20x slower)"
        ),
        RuntimeWarning,
        stacklevel=2,
    )

    def numpy_argmax_reduceat(array, indices):

        array = np.asarray(array)
        indices = np.asarray(indices)

        # splits = np.r_[indices, len(array)]
        splits = np.append(indices, len(array))

        global_indices = np.empty(len(indices), dtype=np.int64)
        for n, (i, j) in enumerate(zip(splits[:-1], splits[1:])):

            if i >= j:
                global_indices[n] = i
                continue

            k = np.argmax(array[i:j])
            global_indices[n] = k + i

        return global_indices

    def numpy_argmin_reduceat(array, indices):

        array = np.asarray(array)
        indices = np.asarray(indices)

        splits = np.r_[indices, len(array)]

        global_indices = []
        for i, j in zip(splits[:-1], splits[1:]):

            if i >= j:
                global_indices.append(i)
                continue

            k = np.argmin(array[i:j])
            global_indices.append(k + i)

        return np.array(global_indices)


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
    Counts the number of elements along segments x[s_ind:e_ind]

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
    TODO: add docstring
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
    Returns:
        - max values within each segment
        - (optional) indices of these max values in the original array
    """
    indices = segmented_argmax(x, s_ind, e_ind)

    if return_indices:
        return x[indices], indices
    return x[indices]


def segmented_argmin(x, s_ind, e_ind):

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
        (True) only full-sized windows are returned.
        (False) partial windows near the boundaries are included as well.

    Returns
    -------
    s_ind : np.ndarray of int
        The array of start indices for each window.
    e_ind : np.ndarray of int
        The array of end indices for each window (exclusive).

    Examples
    --------
    >>> compute_sliding_window_indices(5, 3, same_size=True)
    s_ind: [0, 1, 2]
    e_ind: [3, 4, 5]

    >>> compute_sliding_window_indices(5, 4, same_size=True)
    s_ind: [0, 1]
    e_ind: [4, 5]

    >>> compute_sliding_window_indices(5, 6, same_size=True)
    s_ind: []
    e_ind: []

    >>> compute_sliding_window_indices(5, 3, same_size=False)
    s_ind:           [0, 0, 1, 2, 3]
    e_ind:           [2, 3, 4, 5, 5]
    -> segment_size: [2, 3, 3, 3, 2]

    >>> compute_sliding_window_indices(5, 4, same_size=False)
    s_ind:           [0, 0, 0, 1, 2, 3]
    e_ind:           [2, 3, 4, 5, 5, 5]
    -> segment_size: [2, 3, 4, 4, 3, 2]

    >>> compute_sliding_window_indices(5, 6, same_size=False)
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

    func_map = {
        "sum": segmented_sum,
        "count": segmented_count,
        "mean": segmented_mean,
        "max": segmented_max,
        "min": segmented_min,
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
