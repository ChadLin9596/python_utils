#include "common.hpp"

template <typename T>
static inline npy_intp argmin_segment(const T* a, npy_intp beg, npy_intp end) {

    npy_intp best_i = beg;
    T best = a[beg];

    for (npy_intp i = beg + 1; i < end; ++i) {

        const T v = a[i];
        if (v >= best) continue;

        best = v;
        best_i = i;
    }
    return best_i;
}

template <typename T>
static void argmin_reduceat_core(const T* a,
                                 const npy_intp* idxp,
                                 npy_intp idx_len,
                                 npy_intp arr_len,
                                 npy_intp* outp)
{
    for (npy_intp k = 0; k < idx_len; ++k) {
        const npy_intp beg = idxp[k];
        const npy_intp end = (k + 1 < idx_len) ? idxp[k + 1] : arr_len;
        outp[k] = (end <= beg) ? beg : argmin_segment<T>(a, beg, end);
    }
}

using core_fn = void (*)(const void* a_void,
                         const npy_intp* idxp,
                         npy_intp idx_len,
                         npy_intp arr_len,
                         npy_intp* outp);

template <typename T>
static void core_trampoline(const void* a_void,
                            const npy_intp* idxp,
                            npy_intp idx_len,
                            npy_intp arr_len,
                            npy_intp* outp)
{
    const T* a = static_cast<const T*>(a_void);
    argmin_reduceat_core<T>(a, idxp, idx_len, arr_len, outp);
}

static PyObject* argmin_reduceat_impl(PyObject* a_obj, PyObject* idx_obj) {

    const int req = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;

    // casting a_obj to PyArrayObject a_arr
    // if conversion failed, return NULL
    PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_NOTYPE, req);
    if (!arr) { return NULL; }

    // casting idx_obj to PyArrayObject idx_arr,
    // if conversion failed, decreasing reference count for a_arr for deallocation
    PyArrayObject *idx = (PyArrayObject*)PyArray_FROM_OTF(idx_obj, NPY_INTP, req);
    if (!idx) { Py_DECREF(arr); return NULL; }

    // Check if a_arr is 1D non-empty array
    if (PyArray_NDIM(arr) != 1 || PyArray_DIM(arr, 0) == 0) {
        Py_DECREF(arr);
        Py_DECREF(idx);
        PyErr_SetString(PyExc_ValueError, "a must be 1D non-empty array");
        return NULL;
    }

    // Check if idx_arr is 1D non-empty array
    if (PyArray_NDIM(idx) != 1 || PyArray_DIM(idx, 0) == 0) {
        Py_DECREF(arr);
        Py_DECREF(idx);
        PyErr_SetString(PyExc_ValueError, "idx must be 1D non-empty array");
        return NULL;
    }

    const npy_intp arr_len = PyArray_DIM(arr, 0);
    const npy_intp idx_len = PyArray_DIM(idx, 0);

    const char *msg = NULL;
    const npy_intp* idxp = reinterpret_cast<const npy_intp*>(PyArray_DATA(idx));
    if (check_idx_bounds(idxp, idx_len, arr_len, &msg) != 0) {
        Py_DECREF(arr);
        Py_DECREF(idx);
        PyErr_SetString(PyExc_IndexError, msg);
        return NULL;
    }

    // create output array
    npy_intp odims[1] = { idx_len };
    PyArrayObject *out = (PyArrayObject*)PyArray_SimpleNew(1, odims, NPY_INTP);
    if (!out) {
        Py_DECREF(arr);
        Py_DECREF(idx);
        return NULL;
    }

    // Pointers
    const void* a_data = PyArray_DATA(arr);  // void*; we'll cast in the trampoline
    npy_intp* outp = reinterpret_cast<npy_intp*>(PyArray_DATA(out));

    // dtype dispatch BEFORE releasing GIL
    core_fn run = nullptr;
    const int typenum = PyArray_TYPE(arr);
    switch (typenum) {
        case NPY_FLOAT128: run = &core_trampoline<npy_float128>; break;
        case NPY_FLOAT64: run = &core_trampoline<npy_float64>; break;
        case NPY_FLOAT32: run = &core_trampoline<npy_float32>; break;
        case NPY_FLOAT16: run = &core_trampoline<npy_float16>; break;
        case NPY_INT64:   run = &core_trampoline<npy_int64>;   break;
        case NPY_INT32:   run = &core_trampoline<npy_int32>;   break;
        case NPY_INT16:   run = &core_trampoline<npy_int16>;   break;
        case NPY_INT8:    run = &core_trampoline<npy_int8>;    break;
        case NPY_UINT64:  run = &core_trampoline<npy_uint64>;  break;
        case NPY_UINT32:  run = &core_trampoline<npy_uint32>;  break;
        case NPY_UINT16:  run = &core_trampoline<npy_uint16>;  break;
        case NPY_UINT8:   run = &core_trampoline<npy_uint8>;   break;
        default:
            Py_DECREF(arr); Py_DECREF(idx); Py_DECREF(out);
            PyErr_SetString(PyExc_TypeError, "Unsupported dtype for argmin_reduceat");
            return NULL;
    }

    // run typed kernel without the GIL
    NPY_BEGIN_ALLOW_THREADS
    run(a_data, idxp, idx_len, arr_len, outp);
    NPY_END_ALLOW_THREADS

    Py_DECREF(arr);
    Py_DECREF(idx);
    return (PyObject*)out;
}

// ---- Python glue ----
static PyObject* py_argmin_reduceat(PyObject* self, PyObject* args) {
    PyObject *a_obj, *idx_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &idx_obj)) return NULL;
    return argmin_reduceat_impl(a_obj, idx_obj);
}

// Method table
static PyMethodDef Methods[] = {
    {"reduceat", (PyCFunction)py_argmin_reduceat, METH_VARARGS,
     "Argmin per ufunc.reduceat semantics (global indices)."},
    {NULL, NULL, 0, NULL}
};

// Module Definition (Py3)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "argmin",           // module name
    "argmin reduceat",  // module docstring
    -1,                 // module size
    Methods             // module method
};

// PyInit_<module_name>
PyMODINIT_FUNC PyInit_argmin(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
