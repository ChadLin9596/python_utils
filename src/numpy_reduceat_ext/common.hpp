#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>


static inline int check_idx_bounds(const npy_intp *idx, npy_intp m, npy_intp n, const char **errmsg) {
    for (npy_intp k = 0; k < m; ++k) {
        if (idx[k] < 0 || idx[k] >= n) {
            *errmsg = "idx elements must be within [0, len(a))";
            return -1;
        }
    }
    return 0;
}
