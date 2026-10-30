import unittest

import numpy as np

from scripts.paper_statistics import geometry


class GeometryTests(unittest.TestCase):
    def test_orthogonal_points_have_unit_spread_and_diagonal_centroid(self):
        units = np.eye(2)
        direction = np.array([1., 0.])
        distance, spread = geometry(units, direction)
        self.assertAlmostEqual(distance, 1 - 1 / np.sqrt(2))
        self.assertAlmostEqual(spread, 1.)
        rotation = np.array([[0., -1.], [1., 0.]])
        np.testing.assert_allclose(geometry(units @ rotation, direction @ rotation), (distance, spread))
