import unittest
import numpy as np
import pandas as pd
from riemann_stats_py import RiemannianUMAPAnalysis, Visualization, DataProcessing, pca_inertia_by_components  # :contentReference[oaicite:0]{index=0}

class TestRiemannianUMAPAnalysis(unittest.TestCase):

    def setUp(self):
        # Se crea un DataFrame pequeño de 3 muestras y 2 características
        self.data = pd.DataFrame({
            'a': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
            'b': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        })
        # Instanciamos la clase con n_neighbors=2 (para simplificar)
        self.analysis = RiemannianUMAPAnalysis(self.data, n_neighbors=2)

    def test_calculate_umap_graph_similarities(self):
        sim_matrix = self.analysis.calculate_umap_graph_similarities()
        # Verificamos que se retorne un numpy.ndarray de forma (n, n)
        n = self.data.shape[0]
        self.assertIsInstance(sim_matrix, np.ndarray)
        self.assertEqual(sim_matrix.shape, (n, n), "La matriz de similitudes debe ser de forma (n, n)")

    def test_calculate_rho_matrix(self):
        sim_matrix = self.analysis.calculate_umap_graph_similarities()
        rho_matrix = self.analysis.calculate_rho_matrix()
        # Verificamos que cada elemento de rho sea 1 menos el correspondiente de sim_matrix
        np.testing.assert_allclose(rho_matrix, 1 - sim_matrix)

    def test_riemannian_vector_difference(self):
        # Se requiere haber calculado primero las similitudes y la matriz rho
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        riemann_diff = self.analysis.riemannian_vector_difference()
        n = self.data.shape[0]
        features = self.data.shape[1]
        self.assertEqual(riemann_diff.shape, (n, n, features),
                         "El array de diferencias riemannianas debe tener forma (n, n, features)")
        # Verificamos el valor para un par (ej. i=0, j=1)
        expected = self.analysis.rho[0, 1] * (self.data.iloc[0] - self.data.iloc[1]).values
        np.testing.assert_allclose(riemann_diff[0, 1], expected)

    def test_calculate_umap_distance_matrix(self):
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        dist_matrix = self.analysis.calculate_umap_distance_matrix()
        n = self.data.shape[0]
        self.assertEqual(dist_matrix.shape, (n, n), "La matriz de distancias debe tener forma (n, n)")
        # Para un elemento, comprobamos que la norma del vector de diferencia coincide
        diff = self.analysis.riemannian_diff[0, 1]
        expected_norm = np.linalg.norm(diff)
        self.assertAlmostEqual(dist_matrix[0, 1], expected_norm, places=5)

    def test_riemannian_covariance_matrix(self):
        # Se deben haber calculado previamente las matrices de distancia y diferencias
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        cov_matrix = self.analysis.riemannian_covariance_matrix()
        features = self.data.shape[1]
        self.assertEqual(cov_matrix.shape, (features, features),
                         "La matriz de covarianza debe ser cuadrada de dimensión (features, features)")

    def test_riemannian_covariance_matrix_general(self):
        # Se calcula una matriz de covarianza general para un DataFrame combinado
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        # Se concatenan datos originales con una matriz de componentes dummy
        dummy_components = pd.DataFrame({
            'comp1': [0.1, 0.2, 0.3],
            'comp2': [0.4, 0.5, 0.6]
        })
        combined_data = pd.concat([self.data, dummy_components], axis=1)
        cov_matrix_general = self.analysis.riemannian_covariance_matrix_general(combined_data)
        n_features = combined_data.shape[1]
        self.assertEqual(cov_matrix_general.shape, (n_features, n_features),
                         "La matriz de covarianza general debe ser de dimensión (n_features, n_features)")

    def test_riemannian_correlation_matrix(self):
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        corr_matrix = self.analysis.riemannian_correlation_matrix()
        features = self.data.shape[1]
        self.assertEqual(corr_matrix.shape, (features, features),
                         "La matriz de correlación debe ser de dimensión (features, features)")
        # Se verifica que la diagonal sea aproximadamente 1
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
        # Se espera que la matriz de componentes tenga n filas y "features" columnas
        self.assertEqual(components.shape, (n, features),
                         "La matriz de componentes debe tener forma (n, features)")

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
                         "La función riemannian_components debe retornar una matriz de forma (n, features)")

    def test_riemannian_correlation_variables_components(self):
        # Se necesita que la matriz de componentes tenga al menos dos columnas
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()
        corr_matrix = self.analysis.riemannian_correlation_matrix()
        components = self.analysis.riemannian_components_from_data_and_correlation(corr_matrix)
        correlations_df = self.analysis.riemannian_correlation_variables_components(components)
        # El DataFrame debe tener una fila por cada característica y columnas "Component_1" y "Component_2"
        self.assertEqual(correlations_df.shape[0], self.data.shape[1],
                         "El número de filas del DataFrame debe ser igual al número de características")
        self.assertListEqual(list(correlations_df.columns), ["Component_1", "Component_2"],
                             "Las columnas deben ser ['Component_1', 'Component_2']")

if __name__ == '__main__':
    unittest.main()
