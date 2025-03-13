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

    def covariance_matrix(self):
        """
        Calcula la matriz de covarianza clásica.
        """
        centered_data = self.data - np.mean(self.data, axis=0)
        return np.dot(centered_data.T, centered_data) / self.data.shape[0]

    def riemannian_covariance_matrix(self):
        """
        Calcula la matriz de covarianza utilizando diferencias Riemannianas.
        """
        if self.umap_distance_matrix is None:
            raise ValueError("Debe calcular la matriz de distancias UMAP antes de obtener la matriz de covarianza Riemanniana.")
        riemannian_mean_index = np.argmin(np.sum(self.umap_distance_matrix, axis=1))
        n_samples, n_features = self.data.shape
        cov_matrix = np.zeros((n_features, n_features))
        for i in range(n_samples):
            diff_vector = self.rho[i, riemannian_mean_index] * (self.data.iloc[i] - self.data.iloc[riemannian_mean_index])
            cov_matrix += np.outer(diff_vector, diff_vector)
        return cov_matrix / n_samples

    def correlation_matrix(self, cov_matrix):
        """
        Calcula la matriz de correlación a partir de una matriz de covarianza dada.
        """
        n = cov_matrix.shape[0]
        corr_matrix = np.zeros_like(cov_matrix)
        for i in range(n):
            for j in range(n):
                corr_matrix[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
        return corr_matrix

    def principal_components(self, corr_matrix):
        """
        Calcula los componentes principales a partir de la matriz de correlación.
        """
        mean_centered_data = self.data - np.mean(self.data, axis=0)
        std_population = np.sqrt(np.sum(mean_centered_data ** 2, axis=0) / self.data.shape[0])
        standardized_data = mean_centered_data / std_population
        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        return np.dot(standardized_data, eigenvectors[:, sorted_indices])
