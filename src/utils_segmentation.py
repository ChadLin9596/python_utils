import numpy as np


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


def segmented_max(x, s_ind, e_ind, return_indices=False):
    """
    Computes the maximum along segments x[s_ind:e_ind, ...]
    Returns:
        - max values within each segment
        - (optional) indices of these max values in the original array
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

    results = np.maximum.reduceat(x, splits)[::2]

    if return_indices:
        max_ind = segmented_where(x, s_ind, e_ind, results)
        return results, max_ind
    return results


def segmented_min(x, s_ind, e_ind, return_indices=False):
    """
    Computes the minimum along segments x[s_ind:e_ind, ...]
    Returns:
        - min values within each segment
        - (optional) indices of these min values in the original array
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

    results = np.minimum.reduceat(x, splits)[::2]

    if return_indices:
        min_ind = segmented_where(x, s_ind, e_ind, results)
        return results, min_ind
    return results
