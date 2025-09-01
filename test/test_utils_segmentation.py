import unittest

import numpy as np

import py_utils.utils_segmentation as utils_segmentation


class TestSegmentedOperations(unittest.TestCase):

    def test_segmented_sum(self):
        x = np.array([1, 2, 3, 4, 5])
        s_ind = np.array([0, 3])
        e_ind = np.array([3, 5])
        expected = np.array([6, 9])  # 1+2+3 and 4+5

        np.testing.assert_array_equal(
            utils_segmentation.segmented_sum(x, s_ind, e_ind), expected
        )

    def test_segmented_count(self):
        x = np.array([1, 2, 3, 4, 5])
        s_ind = np.array([0, 3])
        e_ind = np.array([3, 5])

        # Three elements in first segment, two in second
        expected = np.array([3, 2])

        np.testing.assert_array_equal(
            utils_segmentation.segmented_count(x, s_ind, e_ind), expected
        )

    def test_segmented_mean(self):
        x = np.array([1, 2, 3, 4, 5])
        s_ind = np.array([0, 3])
        e_ind = np.array([3, 5])

        # Mean of 1,2,3 and 4,5
        expected = np.array([2, 4.5])

        np.testing.assert_array_equal(
            utils_segmentation.segmented_mean(x, s_ind, e_ind), expected
        )

    def test_segmented_max(self):
        x = np.array([2, 3, 5, 1, 4, 7])
        s_ind = np.array([0, 2, 3, 5])
        e_ind = np.array([2, 3, 6, 6])

        expected = np.array([3, 5, 7, 7])

        np.testing.assert_array_equal(
            utils_segmentation.segmented_max(x, s_ind, e_ind), expected
        )

    def test_segmented_min(self):
        x = np.array([2, 3, 5, 1, 4, 7])
        s_ind = np.array([0, 2, 3, 5])
        e_ind = np.array([2, 3, 6, 6])

        expected = np.array([2, 5, 1, 7])

        np.testing.assert_array_equal(
            utils_segmentation.segmented_min(x, s_ind, e_ind), expected
        )

    def test_segmented_where(self):
        x = np.array([2, 3, 5, 1, 4, 7])
        s_ind = np.array([0, 2, 3, 5])
        e_ind = np.array([2, 3, 6, 6])
        value = np.array([3, 5, 7, 7])

        expected = [
            np.array([1]),
            np.array([2]),
            np.array([5]),
            np.array([5]),
        ]
        results = utils_segmentation.segmented_where(x, s_ind, e_ind, value)

        self.assertEqual(len(results), len(expected))
        for i in range(len(results)):
            np.testing.assert_array_equal(results[i], expected[i])

    def test_segmented_max_2(self):
        x = np.array([2, 3, 5, 1, 4, 7])
        s_ind = np.array([0, 2, 3, 5])
        e_ind = np.array([2, 3, 6, 6])

        expected = np.array([3, 5, 7, 7])
        expected_indices = [
            np.array([1]),
            np.array([2]),
            np.array([5]),
            np.array([5]),
        ]
        results, indices = utils_segmentation.segmented_max(
            x,
            s_ind,
            e_ind,
            return_indices=True,
        )

        np.testing.assert_array_equal(results, expected)
        self.assertEqual(len(indices), len(expected_indices))
        for i in range(len(indices)):
            np.testing.assert_array_equal(indices[i], expected_indices[i])

    def test_segmented_min_2(self):
        x = np.array([2, 3, 5, 1, 4, 7])
        s_ind = np.array([0, 2, 3, 5])
        e_ind = np.array([2, 3, 6, 6])

        expected = np.array([2, 5, 1, 7])
        expected_indices = [
            np.array([0]),
            np.array([2]),
            np.array([3]),
            np.array([5]),
        ]
        results, indices = utils_segmentation.segmented_min(
            x,
            s_ind,
            e_ind,
            return_indices=True,
        )

        np.testing.assert_array_equal(results, expected)
        self.assertEqual(len(indices), len(expected_indices))
        for i in range(len(indices)):
            np.testing.assert_array_equal(indices[i], expected_indices[i])

    def test_segmented_max_3(self):
        x = np.array([2, 3, 3, 5, 1, 1, 4, 7])
        s_ind = np.array([0, 0, 1, 3, 3, 7])
        e_ind = np.array([2, 3, 3, 7, 8, 8])

        expected = np.array([3, 3, 3, 5, 7, 7])
        expected_indices = np.array([1, 1, 1, 3, 7, 7])

        results, indices = utils_segmentation.segmented_max(
            x,
            s_ind,
            e_ind,
            return_indices=True,
        )

        np.testing.assert_array_equal(results, expected)
        self.assertEqual(len(indices), len(expected_indices))
        for i in range(len(indices)):
            np.testing.assert_array_equal(indices[i], expected_indices[i])

    def test_segmented_min_3(self):
        x = np.array([2, 3, 3, 5, 1, 1, 4, 7])
        s_ind = np.array([0, 0, 1, 3, 3, 7])
        e_ind = np.array([2, 3, 3, 7, 8, 8])

        expected = np.array([2, 2, 3, 1, 1, 7])
        expected_indices = np.array([0, 0, 1, 4, 4 ,7])

        results, indices = utils_segmentation.segmented_min(
            x,
            s_ind,
            e_ind,
            return_indices=True,
        )

        np.testing.assert_array_equal(results, expected)
        self.assertEqual(len(indices), len(expected_indices))
        for i in range(len(indices)):
            np.testing.assert_array_equal(indices[i], expected_indices[i])

    def test_error_conditions(self):
        x = np.array([1, 2, 3])
        s_ind = np.array([0, 2])
        e_ind = np.array([2, 4])  # Invalid e_ind

        with self.assertRaises(ValueError):
            utils_segmentation.segmented_sum(x, s_ind, e_ind)

        with self.assertRaises(ValueError):
            utils_segmentation.segmented_count(x, s_ind, e_ind)

        with self.assertRaises(ValueError):
            utils_segmentation.segmented_mean(x, s_ind, e_ind)


if __name__ == "__main__":
    unittest.main()
