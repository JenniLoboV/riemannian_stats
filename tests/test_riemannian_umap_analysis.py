import unittest
import numpy as np
import pandas as pd
from riemann_stats_py import RiemannianUMAPAnalysis

class TestRiemannianUMAPAnalysis(unittest.TestCase):

    def setUp(self):
        # Create a small DataFrame with 10 samples and 2 features
        self.data = pd.DataFrame({
            'a': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
            'b': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        })
        # Instantiate the class with n_neighbors=2 (to simplify)
        self.analysis = RiemannianUMAPAnalysis(self.data, n_neighbors=2)

    def test_calculate_umap_graph_similarities(self):
        sim_matrix = self.analysis.calculate_umap_graph_similarities()
        # Verify that a numpy.ndarray of shape (n, n) is returned
        n = self.data.shape[0]
        self.assertIsInstance(sim_matrix, np.ndarray)
        self.assertEqual(sim_matrix.shape, (n, n), "The similarity matrix must have shape (n, n)")

    def test_calculate_rho_matrix(self):
        sim_matrix = self.analysis.calculate_umap_graph_similarities()
        rho_matrix = self.analysis.calculate_rho_matrix()
        # Verify that each element of rho equals 1 minus the corresponding element of sim_matrix
        np.testing.assert_allclose(rho_matrix, 1 - sim_matrix)

    def test_riemannian_vector_difference(self):
        # It is required that the similarities and the rho matrix are calculated first
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        riemann_diff = self.analysis.riemannian_vector_difference()
        n = self.data.shape[0]
        features = self.data.shape[1]
        self.assertEqual(riemann_diff.shape, (n, n, features),
                         "The array of Riemannian differences must have shape (n, n, features)")
        # Verify the value for a pair (e.g., i=0, j=1)
        expected = self.analysis.rho[0, 1] * (self.data.iloc[0] - self.data.iloc[1]).values
        np.testing.assert_allclose(riemann_diff[0, 1], expected)

    def test_calculate_umap_distance_matrix(self):
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        dist_matrix = self.analysis.calculate_umap_distance_matrix()
        n = self.data.shape[0]
        self.assertEqual(dist_matrix.shape, (n, n), "The distance matrix must have shape (n, n)")
        # For an element, verify that the norm of the difference vector matches
        diff = self.analysis.riemannian_diff[0, 1]
        expected_norm = np.linalg.norm(diff)
        self.assertAlmostEqual(dist_matrix[0, 1], expected_norm, places=5)

    def test_riemannian_covariance_matrix(self):
        # The distance and difference matrices must be computed beforehand
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        cov_matrix = self.analysis.riemannian_covariance_matrix()
        features = self.data.shape[1]
        self.assertEqual(cov_matrix.shape, (features, features),
                         "The covariance matrix must be square with dimensions (features, features)")

    def test_riemannian_covariance_matrix_general(self):
        # Calculate a general covariance matrix for a combined DataFrame
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        # Concatenate original data with a dummy components matrix
        dummy_components = pd.DataFrame({
            'comp1': [0.1, 0.2, 0.3],
            'comp2': [0.4, 0.5, 0.6]
        })
        combined_data = pd.concat([self.data, dummy_components], axis=1)
        cov_matrix_general = self.analysis.riemannian_covariance_matrix_general(combined_data)
        n_features = combined_data.shape[1]
        self.assertEqual(cov_matrix_general.shape, (n_features, n_features),
                         "The general covariance matrix must have dimensions (n_features, n_features)")

    def test_riemannian_correlation_matrix(self):
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        corr_matrix = self.analysis.riemannian_correlation_matrix()
        features = self.data.shape[1]
        self.assertEqual(corr_matrix.shape, (features, features),
                         "The correlation matrix must have dimensions (features, features)")
        # Verify that the diagonal elements are approximately 1
        for i in range(features):
            self.assertAlmostEqual(corr_matrix[i, i], 1.0, places=5)

    def test_riemannian_components_from_data_and_correlation(self):
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        corr_matrix = self.analysis.riemannian_correlation_matrix()
        components = self.analysis.riemannian_components_from_data_and_correlation(corr_matrix)
        n = self.data.shape[0]
        features = self.data.shape[1]
        # Expect the components matrix to have n rows and "features" columns
        self.assertEqual(components.shape, (n, features),
                         "The components matrix must have shape (n, features)")

    def test_riemannian_components(self):
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        corr_matrix = self.analysis.riemannian_correlation_matrix()
        components = self.analysis.riemannian_components(corr_matrix)
        n = self.data.shape[0]
        features = self.data.shape[1]
        self.assertEqual(components.shape, (n, features),
                         "The riemannian_components function must return a matrix with shape (n, features)")

    def test_riemannian_correlation_variables_components(self):
        # The components matrix must have at least two columns
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        corr_matrix = self.analysis.riemannian_correlation_matrix()
        components = self.analysis.riemannian_components_from_data_and_correlation(corr_matrix)
        correlations_df = self.analysis.riemannian_correlation_variables_components(components)
        # The DataFrame must have one row for each feature and columns "Component_1" and "Component_2"
        self.assertEqual(correlations_df.shape[0], self.data.shape[1],
                         "The number of rows in the DataFrame must equal the number of features")
        self.assertListEqual(list(correlations_df.columns), ["Component_1", "Component_2"],
                             "The columns must be ['Component_1', 'Component_2']")

if __name__ == '__main__':
    unittest.main()
