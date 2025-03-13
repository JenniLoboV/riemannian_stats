import matplotlib

matplotlib.use('TkAgg')  # O puedes probar con 'Agg', 'Qt5Agg', 'GTK3Agg', etc.
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class UMAPAnalysis:
    """
    Clase para realizar análisis basado en UMAP, incluyendo cálculos de similitud,
    matrices de covarianza y correlación, y visualización de resultados.
    """

    def __init__(self, data, n_neighbors=3, min_dist=0.1, metric='euclidean'):
        """
        Inicializa la clase con los parámetros de UMAP y los datos.

        Parameters:
        data (numpy array o DataFrame): Datos de entrada.
        n_neighbors (int): Número de vecinos para UMAP.
        min_dist (float): Distancia mínima en el espacio reducido.
        metric (str): Métrica de distancia utilizada en UMAP.
        """
        self.data = data
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.umap_similarities = None
        self.rho = None
        self.riemannian_diff = None
        self.umap_distance_matrix = None

    def calculate_umap_graph_similarities(self):
        """
        Calcula las distancias UMAP basadas en la gráfica de conexión KNN.
        """
        reducer = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric)
        reducer.fit(self.data)
        umap_graph = reducer.graph_
        self.umap_similarities = np.array(umap_graph.todense())
        return self.umap_similarities

    def calculate_rho_matrix(self):
        """
        Calcula la matriz Rho como 1 menos las similitudes UMAP.
        """
        if self.umap_similarities is None:
            raise ValueError("Debe calcular las similitudes de UMAP antes de obtener la matriz Rho.")
        self.rho = 1 - self.umap_similarities
        return self.rho

    def riemannian_vector_difference(self):
        """
        Calcula la diferencia Riemanniana entre cada par de vectores fila en la matriz de datos.
        """
        if self.rho is None:
            raise ValueError("Debe calcular la matriz Rho antes de calcular diferencias Riemannianas.")
        n_rows = self.data.shape[0]
        self.riemannian_diff = np.zeros((n_rows, n_rows, self.data.shape[1]))
        for i in range(n_rows):
            for j in range(n_rows):
                self.riemannian_diff[i, j] = self.rho[i, j] * (self.data.iloc[i] - self.data.iloc[j])
        return self.riemannian_diff

    def calculate_umap_distance_matrix(self):
        """
        Calcula la matriz de distancias UMAP utilizando diferencias Riemannianas ponderadas.
        """
        if self.riemannian_diff is None:
            raise ValueError("Debe calcular la diferencia Riemanniana antes de obtener la matriz de distancias UMAP.")
        n_rows = self.riemannian_diff.shape[0]
        self.umap_distance_matrix = np.zeros((n_rows, n_rows))
        for i in range(n_rows):
            for j in range(n_rows):
                self.umap_distance_matrix[i, j] = np.linalg.norm(self.riemannian_diff[i, j])
        return self.umap_distance_matrix

    def riemannian_covariance_matrix(self):
        """
        Calcula la matriz de covarianza utilizando diferencias Riemannianas.
        """
        if self.umap_distance_matrix is None:
            raise ValueError(
                "Debe calcular la matriz de distancias UMAP antes de obtener la matriz de covarianza Riemanniana.")
        riemannian_mean_index = np.argmin(np.sum(self.umap_distance_matrix, axis=1))
        n_samples, n_features = self.data.shape
        cov_matrix = np.zeros((n_features, n_features))
        for i in range(n_samples):
            diff_vector = self.rho[i, riemannian_mean_index] * (
                        self.data.iloc[i] - self.data.iloc[riemannian_mean_index])
            cov_matrix += np.outer(diff_vector, diff_vector)
        return cov_matrix / n_samples

    # NUEVOS MÉTODOS ADAPTADOS PARA ANÁLISIS RIEMANNIANO

    def riemannian_correlation_matrix(self):
        """
        Calcula la matriz de correlación Riemanniana a partir de la matriz de covarianza Riemanniana.

        Returns:
            corr_matrix_riemannian (numpy array): La matriz de correlación Riemanniana.
        """
        cov_matrix_riemannian = self.riemannian_covariance_matrix()
        n = cov_matrix_riemannian.shape[0]
        corr_matrix_riemannian = np.zeros_like(cov_matrix_riemannian)
        for i in range(n):
            for j in range(n):
                corr_matrix_riemannian[i, j] = cov_matrix_riemannian[i, j] / np.sqrt(
                    cov_matrix_riemannian[i, i] * cov_matrix_riemannian[j, j])
        return corr_matrix_riemannian

    def riemannian_components(self, corr_matrix):
        """
        Realiza un análisis de componentes principales (PCA) Riemanniano usando la matriz de correlación suministrada.

        Parameters:
            corr_matrix (numpy array): Matriz de correlación Riemanniana.

        Returns:
            principal_components (numpy array): Matriz de componentes principales.
        """
        # Verificar que la matriz de correlación sea cuadrada
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValueError("La matriz de correlación debe ser cuadrada.")
        if self.data.shape[1] != corr_matrix.shape[0]:
            raise ValueError(
                "El número de columnas en los datos debe coincidir con el tamaño de la matriz de correlación.")

        # Calcular la media Riemanniana y centrar los datos
        riemannian_mean_index = np.argmin(np.sum(self.umap_distance_matrix, axis=1))
        riemannian_mean_centered_data = np.zeros_like(self.data)
        for i in range(self.data.shape[0]):
            riemannian_mean_centered_data[i] = self.rho[i, riemannian_mean_index] * (
                    self.data.iloc[i] - self.data.iloc[riemannian_mean_index])
        # Calcular la desviación estándar Riemanniana
        riemannian_std_population = np.sqrt(np.sum(riemannian_mean_centered_data ** 2, axis=0) / self.data.shape[0])
        # Estandarizar los datos
        standardized_data = riemannian_mean_centered_data / riemannian_std_population
        # Calcular eigenvalores y eigenvectores
        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        principal_components = np.dot(standardized_data, eigenvectors)
        return principal_components

    def _riemannian_covariance_matrix_general(self, combined_data):
        """
        Método auxiliar para calcular la matriz de covarianza Riemanniana para un conjunto de datos genérico.

        Parameters:
            combined_data (DataFrame): Datos combinados (por ejemplo, datos originales y componentes).

        Returns:
            cov_matrix (numpy array): Matriz de covarianza Riemanniana.
        """
        riemannian_mean_index = np.argmin(np.sum(self.umap_distance_matrix, axis=1))
        n_samples, n_features = combined_data.shape
        cov_matrix = np.zeros((n_features, n_features))
        for i in range(n_samples):
            diff_vector = self.rho[i, riemannian_mean_index] * (
                        combined_data.iloc[i] - combined_data.iloc[riemannian_mean_index])
            cov_matrix += np.outer(diff_vector, diff_vector)
        return cov_matrix / n_samples

    def riemannian_correlation_variables_components(self, components):
        """
        Calcula la correlación (Riemanniana) entre las variables originales y las dos primeras componentes.

        Parameters:
            components (numpy array): Matriz de componentes (se esperan al menos dos columnas).

        Returns:
            correlations (pandas DataFrame): DataFrame con la correlación de cada variable original con la
                                              primera y segunda componente.
        """
        # Combinar los datos originales y las dos primeras componentes en un DataFrame
        combined_data = pd.DataFrame(
            np.hstack((self.data, components[:, 0:2])),
            columns=[f'feature_{i + 1}' for i in range(self.data.shape[1])] + ['Component_1', 'Component_2']
        )
        # Calcular la matriz de covarianza Riemanniana para los datos combinados
        riemannian_cov_matrix = self._riemannian_covariance_matrix_general(combined_data)

        # Inicializar DataFrame para las correlaciones
        correlations = pd.DataFrame(
            index=[f'feature_{i + 1}' for i in range(self.data.shape[1])],
            columns=['Component_1', 'Component_2']
        )
        # Calcular correlaciones para la primera componente
        for i in range(self.data.shape[1]):
            correlations.loc[f'feature_{i + 1}', 'Component_1'] = riemannian_cov_matrix[i, -2] / np.sqrt(
                riemannian_cov_matrix[i, i] * riemannian_cov_matrix[-2, -2])
        # Calcular correlaciones para la segunda componente
        for i in range(self.data.shape[1]):
            correlations.loc[f'feature_{i + 1}', 'Component_2'] = riemannian_cov_matrix[i, -1] / np.sqrt(
                riemannian_cov_matrix[i, i] * riemannian_cov_matrix[-1, -1])
        return correlations
