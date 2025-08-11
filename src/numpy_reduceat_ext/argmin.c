#include "common.h"

static PyObject* argmin_reduceat_impl(PyObject* a_obj, PyObject* idx_obj) {
    if (PyArray_API == NULL) import_array();

    int a_req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
    PyArrayObject *a_arr = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_NOTYPE, a_req);
    if (!a_arr) return NULL;

    int atype = PyArray_TYPE(a_arr);
    if (atype != NPY_FLOAT64 && atype != NPY_INT64) {
        PyArrayObject *tmp = (PyArrayObject*)PyArray_Cast(a_arr, NPY_FLOAT64);
        Py_DECREF(a_arr);
        a_arr = tmp;
        if (!a_arr) return NULL;
        atype = NPY_FLOAT64;
    }
    if (PyArray_NDIM(a_arr) != 1) {
        Py_DECREF(a_arr);
        PyErr_SetString(PyExc_ValueError, "a must be 1D");
        return NULL;
    }

    int idx_req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
    PyArrayObject *idx_arr = (PyArrayObject*)PyArray_FROM_OTF(idx_obj, NPY_INTP, idx_req);
    if (!idx_arr) { Py_DECREF(a_arr); return NULL; }
    if (PyArray_NDIM(idx_arr) != 1) {
        Py_DECREF(a_arr); Py_DECREF(idx_arr);
        PyErr_SetString(PyExc_ValueError, "idx must be 1D");
        return NULL;
    }

    npy_intp n = PyArray_DIM(a_arr, 0);
    npy_intp m = PyArray_DIM(idx_arr, 0);

    npy_intp odims[1] = { m };
    PyArrayObject *out = (PyArrayObject*)PyArray_SimpleNew(1, odims, NPY_INT64);
    if (!out) { Py_DECREF(a_arr); Py_DECREF(idx_arr); return NULL; }

    if (m == 0) { Py_DECREF(a_arr); Py_DECREF(idx_arr); return (PyObject*)out; }
    if (n == 0) {
        Py_DECREF(a_arr); Py_DECREF(idx_arr); Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError, "a is empty but idx is not");
        return NULL;
    }

    const char *errmsg = NULL;
    if (check_idx_bounds((npy_intp*)PyArray_DATA(idx_arr), m, n, &errmsg) != 0) {
        Py_DECREF(a_arr); Py_DECREF(idx_arr); Py_DECREF(out);
        PyErr_SetString(PyExc_IndexError, errmsg);
        return NULL;
    }

    char *a_data = PyArray_BYTES(a_arr);
    npy_intp a_stride = PyArray_STRIDES(a_arr)[0];
    npy_intp *idx = (npy_intp*)PyArray_DATA(idx_arr);
    long long *outp = (long long*)PyArray_DATA(out);

    NPY_BEGIN_ALLOW_THREADS

    if (atype == NPY_INT64) {
        for (npy_intp k = 0; k < m; ++k) {
            npy_intp i = idx[k];
            npy_intp j = (k + 1 < m) ? idx[k + 1] : n;
            if (j <= i) { outp[k] = (long long)i; continue; }

            npy_int64 best_v;
            memcpy(&best_v, a_data + i * a_stride, sizeof(npy_int64));
            npy_intp best_i = i;
            for (npy_intp p = i + 1; p < j; ++p) {
                npy_int64 v;
                memcpy(&v, a_data + p * a_stride, sizeof(npy_int64));
                if (v < best_v) { best_v = v; best_i = p; }
            }
            outp[k] = (long long)best_i;
        }
    } else { // float64
        for (npy_intp k = 0; k < m; ++k) {
            npy_intp i = idx[k];
            npy_intp j = (k + 1 < m) ? idx[k + 1] : n;
            if (j <= i) { outp[k] = (long long)i; continue; }

            double best_v;
            memcpy(&best_v, a_data + i * a_stride, sizeof(double));
            npy_intp best_i = i;
            for (npy_intp p = i + 1; p < j; ++p) {
                double v;
                memcpy(&v, a_data + p * a_stride, sizeof(double));
                // NaNs treated as "not better"
                if (v < best_v) { best_v = v; best_i = p; }
            }
            outp[k] = (long long)best_i;
        }
    }

    NPY_END_ALLOW_THREADS

    Py_DECREF(a_arr);
    Py_DECREF(idx_arr);
    return (PyObject*)out;
}

static PyObject* py_argmin_reduceat(PyObject* self, PyObject* args) {
    PyObject *a_obj, *idx_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &idx_obj)) return NULL;
    return argmin_reduceat_impl(a_obj, idx_obj);
}

static PyMethodDef Methods[] = {
    {"argmin_reduceat", py_argmin_reduceat, METH_VARARGS,
     "Argmin per ufunc.reduceat semantics (global indices)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_argmin",   // <<< module name
    "argmin reduceat",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__argmin(void) {  // <<< init symbol must match module’s last component
    import_array();
    return PyModule_Create(&moduledef);
}