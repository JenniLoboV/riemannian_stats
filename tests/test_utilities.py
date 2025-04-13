import unittest
import numpy as np
from riemann_stats_py import Utilities


class TestPCAInertiaByComponents(unittest.TestCase):
    def setUp(self):
        # Define a valid square correlation matrix
        self.valid_corr_matrix = np.array([
            [1.0, 0.8, 0.5],
            [0.8, 1.0, 0.3],
            [0.5, 0.3, 1.0]
        ])

        # Define an invalid (non-square) correlation matrix
        self.invalid_corr_matrix = np.array([
            [1.0, 0.8],
            [0.8, 1.0],
            [0.5, 0.3]
        ])

    def test_valid_components(self):
        # Check if the explained inertia calculation is correct for valid inputs
        explained_inertia = Utilities.pca_inertia_by_components(self.valid_corr_matrix, 0, 1)
        self.assertTrue(0 <= explained_inertia <= 1,
                        "Explained inertia must be between 0 and 1.")

    def test_invalid_corr_matrix(self):
        # Check if function raises an error for non-square correlation matrix
        with self.assertRaises(ValueError):
            Utilities.pca_inertia_by_components(self.invalid_corr_matrix, 0, 1)

    def test_invalid_component_indices(self):
        # Check if function raises an error for invalid component indices
        with self.assertRaises(ValueError):
            Utilities.pca_inertia_by_components(self.valid_corr_matrix, -1, 1)

        with self.assertRaises(ValueError):
            Utilities.pca_inertia_by_components(self.valid_corr_matrix, 0, 3)

    def test_total_inertia_equals_one(self):
        eigenvalues, _ = np.linalg.eig(self.valid_corr_matrix)
        total_inertia = np.sum(eigenvalues)
        selected_inertia = np.sum(eigenvalues)  # suma de todos los eigenvalores

        explained_inertia = selected_inertia / total_inertia
        self.assertAlmostEqual(explained_inertia, 1.0, places=5,
                               msg="Total inertia should sum up approximately to 1.")


if __name__ == '__main__':
    unittest.main()