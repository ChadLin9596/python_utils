import os
import sys

os.chdir(os.path.split(os.path.realpath(__file__))[0])
sys.path.append("..")

import unittest

import pcd


class TestPCD(unittest.TestCase):

    valid_entries = [
        "VERSION",
        "FIELDS",
        "SIZE",
        "TYPE",
        "COUNT",
        "WIDTH",
        "HEIGHT",
        "VIEWPOINT",
        "POINTS",
        "DATA",
    ]

    def test_read(self):
        foo = pcd.read("example.pcd")

        self.assertTrue(foo.shape == (213,))
        self.assertTrue(foo.dtype.names == ("x", "y", "z", "rgb"))
        self.assertTrue(foo.dtype.itemsize == 16)

    def test_read_header(self):

        _, header = pcd.read("example.pcd", return_header=True)

        for entry in self.valid_entries:
            self.assertTrue(entry in header)


if __name__ == "__main__":
    unittest.main()
