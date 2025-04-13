from typing import Union
import matplotlib

matplotlib.use("TkAgg")  # Alternatively, you can try 'Agg', 'Qt5Agg', 'GTK3Agg', etc.
import umap
import pandas as pd
import numpy as np


class RiemannianUMAPAnalysis:
    """
    Class for performing UMAP-based analysis combined with Riemannian geometry.

    Attributes:
        data (numpy.ndarray or pandas.DataFrame): Input data to be analyzed.
        n_neighbors (int): Number of neighbors used for constructing the UMAP KNN graph.
        min_dist (float): Minimum distance parameter for UMAP, controlling the tightness of clusters.
        metric (str): Distance metric used in UMAP.
        umap_similarities (numpy.ndarray, optional): Matrix of similarity values derived from the UMAP KNN graph.
        rho (numpy.ndarray, optional): Matrix computed as 1 minus the UMAP similarity matrix, used for weighting differences.
        riemannian_diff (numpy.ndarray, optional): 3D array containing the weighted Riemannian differences between each pair of data points.
        umap_distance_matrix (numpy.ndarray, optional): Matrix of distances calculated from the Riemannian differences.
    """

    def __init__(self, data: Union[np.ndarray, pd.DataFrame], n_neighbors: int = 3,
                 min_dist: float = 0.1, metric: str = "euclidean") -> None:
        """
        Initializes the RiemannianUMAPAnalysis class with UMAP parameters and input data.

        Parameters:
            data (numpy.ndarray or pandas.DataFrame): Input data.
            n_neighbors (int): Number of neighbors for UMAP. Defaults to 3.
            min_dist (float): Minimum distance in the reduced space. Defaults to 0.1.
            metric (str): Distance metric used in UMAP. Defaults to "euclidean".
        """
        self.data = data
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.umap_similarities: Union[np.ndarray, None] = None
        self.rho: Union[np.ndarray, None] = None
        self.riemannian_diff: Union[np.ndarray, None] = None
        self.umap_distance_matrix: Union[np.ndarray, None] = None

    def calculate_umap_graph_similarities(self) -> np.ndarray:
        """
        Calculates UMAP similarities based on the KNN connectivity graph.

        Returns:
            numpy.ndarray: UMAP similarity matrix derived from the KNN graph.
        """
        reducer = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric)
        reducer.fit(self.data)
        umap_graph = reducer.graph_
        self.umap_similarities = np.array(umap_graph.todense())
        return self.umap_similarities

    def calculate_rho_matrix(self) -> np.ndarray:
        """
        Calculates the Rho matrix as 1 minus the UMAP similarity matrix.

        Returns:
            numpy.ndarray: Rho matrix.

        Raises:
            ValueError: If UMAP similarities have not been calculated.
        """
        if self.umap_similarities is None:
            raise ValueError("UMAP similarities must be calculated before obtaining the Rho matrix.")
        self.rho = 1 - self.umap_similarities
        return self.rho

    def riemannian_vector_difference(self) -> np.ndarray:
        """
        Calculates the Riemannian difference between each pair of row vectors in the data matrix.

        Returns:
            numpy.ndarray: 3D array containing the Riemannian differences for each pair of rows.

        Raises:
            ValueError: If the Rho matrix has not been calculated.
        """
        if self.rho is None:
            raise ValueError("Rho matrix must be calculated before computing Riemannian differences.")
        n_rows = self.data.shape[0]
        self.riemannian_diff = np.zeros((n_rows, n_rows, self.data.shape[1]))
        for i in range(n_rows):
            for j in range(n_rows):
                self.riemannian_diff[i, j] = self.rho[i, j] * (self.data.iloc[i] - self.data.iloc[j])
        return self.riemannian_diff

    def calculate_umap_distance_matrix(self) -> np.ndarray:
        """
        Calculates the UMAP distance matrix using weighted Riemannian differences.

        Returns:
            numpy.ndarray: UMAP distance matrix.

        Raises:
            ValueError: If the Riemannian differences have not been calculated.
        """
        if self.riemannian_diff is None:
            raise ValueError("Riemannian differences must be calculated before obtaining the UMAP distance matrix.")
        n_rows = self.riemannian_diff.shape[0]
        self.umap_distance_matrix = np.zeros((n_rows, n_rows))
        for i in range(n_rows):
            for j in range(n_rows):
                self.umap_distance_matrix[i, j] = np.linalg.norm(self.riemannian_diff[i, j])
        return self.umap_distance_matrix

    def riemannian_covariance_matrix(self) -> np.ndarray:
        """
        Calculates the covariance matrix using Riemannian differences.

        Returns:
            numpy.ndarray: Riemannian covariance matrix.

        Raises:
            ValueError: If the UMAP distance matrix has not been calculated.
        """
        if self.umap_distance_matrix is None:
            raise ValueError(
                "UMAP distance matrix must be calculated before obtaining the Riemannian covariance matrix.")
        riemannian_mean_index = np.argmin(np.sum(self.umap_distance_matrix, axis=1))
        n_samples, n_features = self.data.shape
        cov_matrix = np.zeros((n_features, n_features))
        for i in range(n_samples):
            diff_vector = self.rho[i, riemannian_mean_index] * (
                    self.data.iloc[i] - self.data.iloc[riemannian_mean_index])
            cov_matrix += np.outer(diff_vector, diff_vector)
        return cov_matrix / n_samples

    def riemannian_covariance_matrix_general(self, combined_data: pd.DataFrame) -> np.ndarray:
        """
        Helper method to calculate the Riemannian covariance matrix for a generic dataset.

        Parameters:
            combined_data (pandas.DataFrame): Combined data (e.g., original data and components).

        Returns:
            numpy.ndarray: Riemannian covariance matrix.
        """
        riemannian_mean_index = np.argmin(np.sum(self.umap_distance_matrix, axis=1))
        n_samples, n_features = combined_data.shape
        cov_matrix = np.zeros((n_features, n_features))
        for i in range(n_samples):
            diff_vector = self.rho[i, riemannian_mean_index] * (
                    combined_data.iloc[i] - combined_data.iloc[riemannian_mean_index])
            cov_matrix += np.outer(diff_vector, diff_vector)
        return cov_matrix / n_samples

    def riemannian_correlation_matrix(self) -> np.ndarray:
        """
        Calculates the Riemannian correlation matrix from the Riemannian covariance matrix.

        Returns:
            numpy.ndarray: Riemannian correlation matrix.
        """
        cov_matrix_riemannian = self.riemannian_covariance_matrix()
        n = cov_matrix_riemannian.shape[0]
        corr_matrix_riemannian = np.zeros_like(cov_matrix_riemannian)
        for i in range(n):
            for j in range(n):
                corr_matrix_riemannian[i, j] = cov_matrix_riemannian[i, j] / np.sqrt(
                    cov_matrix_riemannian[i, i] * cov_matrix_riemannian[j, j])
        return corr_matrix_riemannian

    def riemannian_components_from_data_and_correlation(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Performs Riemannian principal component analysis (PCA) using the data and the provided correlation matrix.

        Parameters:
            corr_matrix (numpy.ndarray): Correlation matrix of the variables.

        Returns:
            numpy.ndarray: Matrix of principal components.

        Raises:
            ValueError: If the correlation matrix is not square or if its size does not match the number of data columns.
        """
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValueError("The correlation matrix must be square.")
        if self.data.shape[1] != corr_matrix.shape[0]:
            raise ValueError("The number of columns in the data must match the size of the correlation matrix.")

        riemannian_mean_index = np.argmin(np.sum(self.umap_distance_matrix, axis=1))
        riemannian_mean_centered_data = np.zeros_like(self.data)
        for i in range(self.data.shape[0]):
            riemannian_mean_centered_data[i] = self.rho[i, riemannian_mean_index] * (
                    self.data.iloc[i] - self.data.iloc[riemannian_mean_index])
        riemannian_std_population = np.sqrt(np.sum(riemannian_mean_centered_data ** 2, axis=0) / self.data.shape[0])
        standardized_data = riemannian_mean_centered_data / riemannian_std_population

        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        principal_components = np.dot(standardized_data, eigenvectors)
        return principal_components

    def riemannian_components(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Performs Riemannian principal component analysis (PCA) using the supplied correlation matrix.

        Parameters:
            corr_matrix (numpy.ndarray): Riemannian correlation matrix.

        Returns:
            numpy.ndarray: Matrix of principal components.

        Raises:
            ValueError: If the correlation matrix is not square or if its size does not match the number of data columns.
        """
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValueError("The correlation matrix must be square.")
        if self.data.shape[1] != corr_matrix.shape[0]:
            raise ValueError("The number of columns in the data must match the size of the correlation matrix.")

        riemannian_mean_index = np.argmin(np.sum(self.umap_distance_matrix, axis=1))
        riemannian_mean_centered_data = np.zeros_like(self.data)
        for i in range(self.data.shape[0]):
            riemannian_mean_centered_data[i] = self.rho[i, riemannian_mean_index] * (
                    self.data.iloc[i] - self.data.iloc[riemannian_mean_index])
        riemannian_std_population = np.sqrt(np.sum(riemannian_mean_centered_data ** 2, axis=0) / self.data.shape[0])
        standardized_data = riemannian_mean_centered_data / riemannian_std_population

        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        principal_components = np.dot(standardized_data, eigenvectors)
        return principal_components

    def riemannian_correlation_variables_components(self, components: np.ndarray) -> pd.DataFrame:
        """
        Calculates the Riemannian correlation between the original variables and the first two components.

        Parameters:
            components (numpy.ndarray): Matrix of components (at least two columns are expected).

        Returns:
            pandas.DataFrame: DataFrame with the correlation of each original variable with the first and second components.
        """
        combined_data = pd.DataFrame(
            np.hstack((self.data, components[:, 0:2])),
            columns=[f"feature_{i + 1}" for i in range(self.data.shape[1])] + ["Component_1", "Component_2"]
        )
        riemannian_cov_matrix = self.riemannian_covariance_matrix_general(combined_data)
        correlations = pd.DataFrame(
            index=[f"feature_{i + 1}" for i in range(self.data.shape[1])],
            columns=["Component_1", "Component_2"]
        )
        for i in range(self.data.shape[1]):
            correlations.loc[f"feature_{i + 1}", "Component_1"] = riemannian_cov_matrix[i, -2] / np.sqrt(
                riemannian_cov_matrix[i, i] * riemannian_cov_matrix[-2, -2])
        for i in range(self.data.shape[1]):
            correlations.loc[f"feature_{i + 1}", "Component_2"] = riemannian_cov_matrix[i, -1] / np.sqrt(
                riemannian_cov_matrix[i, i] * riemannian_cov_matrix[-1, -1])
        return correlations
