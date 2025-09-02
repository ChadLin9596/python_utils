import unittest

import numpy as np

import numpy_reduceat_ext

argmin = numpy_reduceat_ext.numpy_argmin_reduceat
argmax = numpy_reduceat_ext.numpy_argmax_reduceat


arr_types = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.float128,
]

split_types = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
]


class TestNumpyReduceatExt(unittest.TestCase):

    # ind: 0, 1, 2, 3, 4, 5, 6, 7, 8
    arr = [2, 3, 3, 5, 1, 1, 4, 7, 7]

    # (0, 2) -> [2, 3] from arr
    # (2, 3) -> [3]
    # (3, 6) -> [5, 1, 1]
    # (6, 8) -> [4, 7]
    s_0 = [0, 2, 3, 6]
    e_0 = [2, 3, 6, 8]

    # (1, 2) -> [3] from arr
    # (0, 3) -> [2, 3, 3]
    # (3, 6) -> [5, 1, 1]
    # (3, 8) -> [5, 1, 1, 4, 7]
    # (0, 8) -> [2, 3, 3, 5, 1, 1, 4, 7]
    s_1 = [1, 0, 3, 3, 0]
    e_1 = [2, 3, 6, 8, 8]

    ans_argmax_0 = [1, 2, 3, 7]
    ans_argmin_0 = [0, 2, 4, 6]

    ans_argmax_1 = [1, 1, 3, 7, 7]
    ans_argmin_1 = [1, 0, 4, 4, 4]

    def test_argmin_0(self):

        overall_inds = np.vstack([self.s_0, self.e_0]).T.flatten()
        results = argmin(self.arr, overall_inds)[::2]
        np.testing.assert_array_equal(results, self.ans_argmin_0)

    def test_argmin_1(self):

        overall_inds = np.vstack([self.s_1, self.e_1]).T.flatten()
        results = argmin(self.arr, overall_inds)[::2]
        np.testing.assert_array_equal(results, self.ans_argmin_1)

    def test_argmax_0(self):

        overall_inds = np.vstack([self.s_0, self.e_0]).T.flatten()
        results = argmax(self.arr, overall_inds)[::2]
        np.testing.assert_array_equal(results, self.ans_argmax_0)

    def test_argmax_1(self):

        overall_inds = np.vstack([self.s_1, self.e_1]).T.flatten()
        results = argmax(self.arr, overall_inds)[::2]
        np.testing.assert_array_equal(results, self.ans_argmax_1)

    def test_argmin_dtype(self):

        for arr_t in arr_types:

            arr = np.array(self.arr, dtype=arr_t)
            for split_t in split_types:

                s_1 = np.array(self.s_1, dtype=split_t)
                e_1 = np.array(self.e_1, dtype=split_t)
                overall_inds = np.vstack([s_1, e_1])
                overall_inds = overall_inds.T.flatten()

                results = argmin(arr, overall_inds)[::2]

                np.testing.assert_array_equal(results, self.ans_argmin_1)

    def test_argmax_dtype(self):

        for arr_t in arr_types:

            arr = np.array(self.arr, dtype=arr_t)
            for split_t in split_types:

                s_1 = np.array(self.s_1, dtype=split_t)
                e_1 = np.array(self.e_1, dtype=split_t)
                overall_inds = np.vstack([s_1, e_1])
                overall_inds = overall_inds.T.flatten()

                results = argmax(arr, overall_inds)[::2]

                np.testing.assert_array_equal(results, self.ans_argmax_1)


if __name__ == "__main__":
    unittest.main()
