import unittest
import numpy as np
import pandas as pd
from riemann_stats_py import RiemannianUMAPAnalysis

class TestRiemannianUMAPAnalysis(unittest.TestCase):

    def setUp(self):
        """
        setUp runs before each test.

        It creates a small DataFrame with 10 samples and 2 features, and instantiates
        a RiemannianUMAPAnalysis object with n_neighbors=2 for simplified testing.
        """
        self.data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        })
        self.analysis = RiemannianUMAPAnalysis(self.data, n_neighbors=2)

    def test_calculate_umap_graph_similarities(self):
        """
        Test that calculate_umap_graph_similarities returns a NumPy array of the expected shape.

        It verifies that the output is a numpy.ndarray with dimensions (n, n),
        where n is the number of samples in the input data.
        """
        sim_matrix = self.analysis.calculate_umap_graph_similarities()
        n = self.data.shape[0]
        self.assertIsInstance(sim_matrix, np.ndarray)
        self.assertEqual(sim_matrix.shape, (n, n), "The similarity matrix must have shape (n, n)")

    def test_result_calculate_umap_graph_similarities(self):
        """
        Test that the similarity matrix computed matches the expected result.

        The expected matrix is defined as a 10x10 array where each element is set
        according to the reference values (e.g. 1.0 where nodes are connected and 0.0 otherwise).
        """
        sim_matrix = self.analysis.calculate_umap_graph_similarities()
        expected_sim_matrix = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])

        # Compare the computed matrix with the expected one using a floating-point tolerance.
        np.testing.assert_allclose(sim_matrix, expected_sim_matrix, rtol=1e-5, atol=1e-5,
                                   err_msg="UMAP similarity matrix does not match the expected values.")

    def test_calculate_rho_matrix(self):
        """
        Test that the rho matrix is correctly computed as 1 minus the similarity matrix.
        """
        sim_matrix = self.analysis.calculate_umap_graph_similarities()
        rho_matrix = self.analysis.calculate_rho_matrix()
        # Verify that each element of rho equals 1 minus the corresponding element of sim_matrix
        np.testing.assert_allclose(rho_matrix, 1 - sim_matrix)

    def test_result_calculate_rho_matrix(self):
        """
        Verifies that the Rho matrix computed matches the expected result from Example 1 out.pdf.

        The expected matrix is defined based on reference values, ensuring that each element
        conforms to the expected 1 - similarity value.
        """
        # Make sure similarities are already calculated
        if self.analysis.umap_similarities is None:
            self.analysis.calculate_umap_graph_similarities()
        rho_matrix = self.analysis.calculate_rho_matrix()

        expected_rho_matrix = np.array([
            [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
            [0., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
            [1., 0., 1., 0., 1., 1., 1., 1., 1., 1.],
            [1., 1., 0., 1., 0., 1., 1., 1., 1., 1.],
            [1., 1., 1., 0., 1., 0., 1., 1., 1., 1.],
            [1., 1., 1., 1., 0., 1., 0., 1., 1., 1.],
            [1., 1., 1., 1., 1., 0., 1., 0., 1., 1.],
            [1., 1., 1., 1., 1., 1., 0., 1., 0., 1.],
            [1., 1., 1., 1., 1., 1., 1., 0., 1., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 0., 1.]
        ])

        np.testing.assert_allclose(rho_matrix, expected_rho_matrix, rtol=1e-5, atol=1e-5,
                                   err_msg="Rho matrix does not match the expected values.")

    def test_riemannian_vector_difference(self):
        """
        Tests that the riemannian_vector_difference returns an array with the correct shape
        and that for a given pair of samples the computed difference is correct.

        The expected difference for a given pair (here indices 0 and 1) is computed by multiplying
        the rho value for that pair with the difference between their original feature vectors.
        """
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

    def test_result_riemannian_vector_difference(self):
        """
        Verifies that the full 3D array of Riemannian vector differences matches the expected result
        from Example 1 out.pdf.

        The expected 3D array is defined as a 10x10x2 array containing the differences between
        pairs of feature vectors, scaled by the corresponding rho values.
        """
        if self.analysis.rho is None:
            self.analysis.calculate_umap_graph_similarities()
            self.analysis.calculate_rho_matrix()
        diff_3d = self.analysis.riemannian_vector_difference()

        expected_diff = np.array([
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.0, -2.0],
                [-3.0, -3.0],
                [-4.0, -4.0],
                [-5.0, -5.0],
                [-6.0, -6.0],
                [-7.0, -7.0],
                [-8.0, -8.0],
                [-9.0, -9.0]
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.0, -2.0],
                [-3.0, -3.0],
                [-4.0, -4.0],
                [-5.0, -5.0],
                [-6.0, -6.0],
                [-7.0, -7.0],
                [-8.0, -8.0]
            ],
            [
                [2.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.0, -2.0],
                [-3.0, -3.0],
                [-4.0, -4.0],
                [-5.0, -5.0],
                [-6.0, -6.0],
                [-7.0, -7.0]
            ],
            [
                [3.0, 3.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.0, -2.0],
                [-3.0, -3.0],
                [-4.0, -4.0],
                [-5.0, -5.0],
                [-6.0, -6.0]
            ],
            [
                [4.0, 4.0],
                [3.0, 3.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.0, -2.0],
                [-3.0, -3.0],
                [-4.0, -4.0],
                [-5.0, -5.0]
            ],
            [
                [5.0, 5.0],
                [4.0, 4.0],
                [3.0, 3.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.0, -2.0],
                [-3.0, -3.0],
                [-4.0, -4.0]
            ],
            [
                [6.0, 6.0],
                [5.0, 5.0],
                [4.0, 4.0],
                [3.0, 3.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.0, -2.0],
                [-3.0, -3.0]
            ],
            [
                [7.0, 7.0],
                [6.0, 6.0],
                [5.0, 5.0],
                [4.0, 4.0],
                [3.0, 3.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-2.0, -2.0]
            ],
            [
                [8.0, 8.0],
                [7.0, 7.0],
                [6.0, 6.0],
                [5.0, 5.0],
                [4.0, 4.0],
                [3.0, 3.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0]
            ],
            [
                [9.0, 9.0],
                [8.0, 8.0],
                [7.0, 7.0],
                [6.0, 6.0],
                [5.0, 5.0],
                [4.0, 4.0],
                [3.0, 3.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        ])

        np.testing.assert_allclose(diff_3d, expected_diff, rtol=1e-5, atol=1e-5,
                                   err_msg="Riemannian vector differences do not match the expected values.")

    def test_calculate_umap_distance_matrix(self):
        """
        Verifies that the UMAP distance matrix is computed correctly.

        The test checks that the output distance matrix has the expected shape (n x n), and that
        for at least one element (here at index [0, 1]), the value equals the Euclidean norm of the
        corresponding difference vector computed in riemannian_vector_difference.
        """
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

    def test_result_calculate_umap_distance_matrix(self):
        # Ensure the 3D differences exist first
        if self.analysis.riemannian_diff is None:
            self.analysis.calculate_umap_graph_similarities()
            self.analysis.calculate_rho_matrix()
            self.analysis.riemannian_vector_difference()

        dist_matrix = self.analysis.calculate_umap_distance_matrix()

        expected_dist_matrix = np.array([
            [0.0, 0.0, 2.82842712, 4.24264069, 5.65685425, 7.07106781, 8.48528137, 9.89949494, 11.3137085, 12.72792206],
            [0.0, 0.0, 0.0, 2.82842712, 4.24264069, 5.65685425, 7.07106781, 8.48528137, 9.89949494, 11.3137085],
            [2.82842712, 0.0, 0.0, 0.0, 2.82842712, 4.24264069, 5.65685425, 7.07106781, 8.48528137, 9.89949494],
            [4.24264069, 2.82842712, 0.0, 0.0, 0.0, 2.82842712, 4.24264069, 5.65685425, 7.07106781, 8.48528137],
            [5.65685425, 4.24264069, 2.82842712, 0.0, 0.0, 0.0, 2.82842712, 4.24264069, 5.65685425, 7.07106781],
            [7.07106781, 5.65685425, 4.24264069, 2.82842712, 0.0, 0.0, 0.0, 2.82842712, 4.24264069, 5.65685425],
            [8.48528137, 7.07106781, 5.65685425, 4.24264069, 2.82842712, 0.0, 0.0, 0.0, 2.82842712, 4.24264069],
            [9.89949494, 8.48528137, 7.07106781, 5.65685425, 4.24264069, 2.82842712, 0.0, 0.0, 0.0, 2.82842712],
            [11.3137085, 9.89949494, 8.48528137, 7.07106781, 5.65685425, 4.24264069, 2.82842712, 0.0, 0.0, 0.0],
            [12.72792206, 11.3137085, 9.89949494, 8.48528137, 7.07106781, 5.65685425, 4.24264069, 2.82842712, 0.0, 0.0]
        ])

        np.testing.assert_allclose(dist_matrix, expected_dist_matrix, rtol=1e-5, atol=1e-5,
                                   err_msg="UMAP distance matrix does not match the expected values.")

    def test_riemannian_covariance_matrix(self):
        """
        Verifies that the computed Riemannian covariance matrix (from the UMAP distance matrix)
        has the correct square shape corresponding to the number of features.
        """
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        cov_matrix = self.analysis.riemannian_covariance_matrix()
        features = self.data.shape[1]
        self.assertEqual(cov_matrix.shape, (features, features),
                         "The covariance matrix must be square with dimensions (features, features)")

    def test_result_riemannian_covariance_matrix(self):
        """
        Verifies the Riemannian covariance matrix matches
        """
        if self.analysis.umap_distance_matrix is None:
            self.analysis.calculate_umap_graph_similarities()
            self.analysis.calculate_rho_matrix()
            self.analysis.riemannian_vector_difference()
            self.analysis.calculate_umap_distance_matrix()

        cov_matrix = self.analysis.riemannian_covariance_matrix()

        expected_cov_matrix = np.array([
            [8.3, 8.3],
            [8.3, 8.3]
        ])

        np.testing.assert_allclose(cov_matrix, expected_cov_matrix, rtol=1e-5, atol=1e-5,
                                   err_msg="Riemannian covariance matrix does not match the expected values.")
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
    def test_result_riemannian_correlation_matrix(self):
        if self.analysis.umap_distance_matrix is None:
            self.analysis.calculate_umap_graph_similarities()
            self.analysis.calculate_rho_matrix()
            self.analysis.riemannian_vector_difference()
            self.analysis.calculate_umap_distance_matrix()

        corr_matrix = self.analysis.riemannian_correlation_matrix()

        expected_corr_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ])

        np.testing.assert_allclose(corr_matrix, expected_corr_matrix, rtol=1e-5, atol=1e-5,
                                   err_msg="Riemannian correlation matrix does not match the expected values.")
    def test_riemannian_components_from_data_and_correlation(self):
        """
        Verifies that the principal components matrix computed from the correlation matrix
        has the correct shape.

        The expected shape is (n, features) where n is the number of samples.
        """
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
    def test_result_riemannian_components_from_data_and_correlation(self):
        """
        Verifies that the principal components
        """
        if self.analysis.umap_distance_matrix is None:
            self.analysis.calculate_umap_graph_similarities()
            self.analysis.calculate_rho_matrix()
            self.analysis.riemannian_vector_difference()
            self.analysis.calculate_umap_distance_matrix()

        corr_matrix = self.analysis.riemannian_correlation_matrix()
        components = self.analysis.riemannian_components_from_data_and_correlation(corr_matrix)

        # If the PDF shows a matrix of shape (10, #features) for the principal components, store it here.
        expected_components = np.array([
            [-1.96352277e+00, 8.81429070e-18],
            [-1.47264208e+00, 3.91283976e-17],
            [-9.81761387e-01, 4.40714535e-18],
            [0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00],
            [9.81761387e-01, -4.40714535e-18],
            [1.47264208e+00, -3.91283976e-17],
            [1.96352277e+00, -8.81429070e-18],
            [2.45440347e+00, 6.74867596e-17]
        ])

        np.testing.assert_allclose(components, expected_components, rtol=1e-5, atol=1e-5,
                                   err_msg="Riemannian components from data and correlation do not match the expected values.")
    def test_riemannian_components(self):
        """
        Verifies that the riemannian_components method returns a matrix with the correct shape.

        The expected shape of the returned components matrix is (n, features),
        where n is the number of samples and features is the number of columns in the data.
        """
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
        """
        Verifies that the DataFrame produced by riemannian_correlation_variables_components has the correct shape and column names.

        The resulting DataFrame should have one row for each feature and exactly two columns:
        "Component_1" and "Component_2".
        """
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

    def test_result_riemannian_correlation_variables_components(self):
        """
        Verifies that the correlation variables DataFrame (correlations between original features and
        the first two principal components) matches the expected table

        The expected DataFrame is constructed with predefined values for "Component_1" and "Component_2"
        for two features, and the test checks that the computed DataFrame matches in both shape and values,
        taking into account floating-point tolerances.
        """
        if self.analysis.umap_distance_matrix is None:
            self.analysis.calculate_umap_graph_similarities()
            self.analysis.calculate_rho_matrix()
            self.analysis.riemannian_vector_difference()
            self.analysis.calculate_umap_distance_matrix()

        corr_matrix = self.analysis.riemannian_correlation_matrix()
        components = self.analysis.riemannian_components_from_data_and_correlation(corr_matrix)
        result_df = self.analysis.riemannian_correlation_variables_components(components)
        result_df = result_df.astype(np.float64)

        expected_data = {
            "Component_1": [1.0, 1.0],
            "Component_2": [0.018034, 0.018034],
        }
        expected_index = ["feature_1", "feature_2"]
        expected_df = pd.DataFrame(expected_data, index=expected_index, dtype=np.float64)

        pd.testing.assert_frame_equal(result_df, expected_df, rtol=1e-5, atol=1e-5,
                                      check_like=True)



if __name__ == '__main__':
    unittest.main()
