import numpy as np
import warnings

try:
    from . import argmin, argmax

    _numpy_argmax_reduceat = argmax.reduceat
    _numpy_argmin_reduceat = argmin.reduceat

except:
    warnings.warn(
        (
            "numpy_reduceat_ext C++ extension not found; "
            "using slower pure-Python fallback. (~20x slower)"
        ),
        RuntimeWarning,
        stacklevel=2,
    )

    def _numpy_argmax_reduceat(array, indices):

        array = np.asarray(array)
        indices = np.asarray(indices)

        splits = np.r_[indices, len(array)]

        global_indices = np.empty(len(indices), dtype=np.int64)
        for n, (i, j) in enumerate(zip(splits[:-1], splits[1:])):

            if i >= j:
                global_indices[n] = i
                continue

            k = np.argmax(array[i:j])
            global_indices[n] = k + i

        return global_indices

    def _numpy_argmin_reduceat(array, indices):

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


__all__ = ["numpy_argmax_reduceat", "numpy_argmin_reduceat"]


def _cast_indices_to_intp(indices):

    if indices.size == 0:
        return np.array([], dtype=np.intp)

    if indices.max() > np.iinfo(np.intp).max:
        raise TypeError("indices dtype cannot be safely cast to platform intp")

    return indices.astype(np.intp, copy=False)


def numpy_argmax_reduceat(array, indices):

    array = np.asarray(array)
    indices = np.asarray(indices)

    if np.isnan(array).any():
        raise ValueError("Input array contains NaN values")

    if not np.can_cast(indices.dtype, np.intp, casting="safe"):
        indices = _cast_indices_to_intp(indices)

    return _numpy_argmax_reduceat(array, indices)


def numpy_argmin_reduceat(array, indices):

    array = np.asarray(array)
    indices = np.asarray(indices)

    if np.isnan(array).any():
        raise ValueError("Input array contains NaN values")

    if not np.can_cast(indices.dtype, np.intp, casting="safe"):
        indices = _cast_indices_to_intp(indices)

    return _numpy_argmin_reduceat(array, indices)
