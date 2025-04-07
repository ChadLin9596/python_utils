import os
import unittest

import py_utils.ply as ply


class TestPLY(unittest.TestCase):

    valid_entries = ["format", "comment", "elements"]
    _path = os.path.dirname(os.path.realpath(__file__))
    _path = os.path.join(_path, "example.ply")

    def test_read(self):
        output = ply.read(self._path)
        vertex = output["vertex"]
        face = output["face"]
        edge = output["edge"]

        self.assertTrue(vertex.shape == (8,))
        self.assertTrue(len(face) == 12)
        self.assertTrue(edge.shape == (5,))

    def test_read_header(self):

        header = ply.parse_header_by_file(self._path)
        for entry in self.valid_entries:
            self.assertTrue(entry in header)

    def test_write(self):

        output = ply.read(self._path)
        vertex = output["vertex"]
        face = output["face"]
        edge = output["edge"]

        ply.write("output.ply", vertex=vertex, face=face, edge=edge)

        new_output = ply.read("output.ply")
        vertex = new_output["vertex"]
        face = new_output["face"]
        edge = new_output["edge"]

        self.assertTrue(vertex.shape == (8,))
        self.assertTrue(len(face) == 12)
        self.assertTrue(edge.shape == (5,))

        os.remove("output.ply")


if __name__ == "__main__":
    unittest.main()
