import numpy as np

class Utilities:
    class Utilities:
        """
        Class for common utility functions in data science projects.

        This class provides static methods for performing common mathematical and
        statistical operations that facilitate data analysis tasks.

        The methods are designed to be used without requiring instantiation,
        which simplifies the integration of these tools into larger data processing
        and analysis workflows.
        """

    @staticmethod
    def pca_inertia_by_components(correlation_matrix: np.ndarray, component1: int, component2: int) -> float:
        """
        Calculates the inertia explained by two specific principal components in a PCA.

        Parameters:
            correlation_matrix (numpy.ndarray): Correlation matrix of the variables.
            component1 (int): Index of the first principal component (based on descending order of explained variance).
            component2 (int): Index of the second principal component (based on descending order of explained variance).

        Returns:
            float: Inertia explained by the selected principal components.

        Raises:
            ValueError: If the correlation matrix is not square or if the component indices are out of range.
        """
        # Verify that the correlation matrix is square.
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("The correlation matrix must be square.")

        # Verify that the components are valid.
        if not (0 <= component1 < correlation_matrix.shape[0]) or not (0 <= component2 < correlation_matrix.shape[0]):
            raise ValueError("The indices of the principal components must be within the valid range.")

        # Compute eigenvalues and eigenvectors.
        eigenvalues, _ = np.linalg.eig(correlation_matrix)

        # Sort eigenvalues in descending order.
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]

        # Calculate the inertia explained by the two selected components.
        total_inertia = np.sum(eigenvalues)
        selected_inertia = eigenvalues[component1] + eigenvalues[component2]
        explained_inertia = selected_inertia / total_inertia

        return explained_inertia
