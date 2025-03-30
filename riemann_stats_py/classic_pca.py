import numpy as np
import pandas as pd

''' NOTA:
Estos metodos van en el archivo riemannian_umap_analysis.py pero en otra clase de PCA clasico.'''
def euclidean_norm(X, Y):
    """
    Calculates the Euclidean norm between two vectors.

    Parameters:
        X (array-like): First vector.
        Y (array-like): Second vector.

    Returns:
        float: Euclidean norm of the difference between X and Y.
    """
    difference = np.array(X) - np.array(Y)
    return np.linalg.norm(difference)

def covariance_matrix(data):
    """
    Calculate the covariance matrix for a given dataset.

    Parameters:
    data (numpy array): The matrix of data points (each row is a data point, and each column is a variable).

    Returns:
    cov_matrix (numpy array): The covariance matrix.
    """
    # Ensure data is a numpy array
    data = np.array(data)

    # Calculate the mean of each column (variable)
    mean_vector = np.mean(data, axis=0)

    # Center the data by subtracting the mean vector from each row
    centered_data = data - mean_vector

    # Calculate the covariance matrix (divide by n-1 for unbiased estimate)
    # cov_matrix = np.dot(centered_data.T, centered_data) / (data.shape[0] - 1)
    cov_matrix = np.dot(centered_data.T, centered_data) / data.shape[0]

    return cov_matrix

def correlation_matrix(cov_matrix):
    """
    Calculate the correlation matrix from a given covariance matrix.

    Parameters:
    cov_matrix (numpy array): The covariance matrix.

    Returns:
    corr_matrix (numpy array): The correlation matrix.
    """
    # Initialize the correlation matrix with zeros
    corr_matrix = np.zeros_like(cov_matrix)

    # Number of variables (features)
    n = cov_matrix.shape[0]

    # Loop over all elements of the covariance matrix to calculate correlations
    for i in range(n):
        for j in range(n):
            # Calculate correlation from covariance
            corr_matrix[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])

    return corr_matrix

def components_from_data_and_correlation(data, correlation_matrix):
    """
    Performs Principal Component Analysis (PCA) using a data table
    and a correlation matrix.

    Args:
        data (numpy.ndarray): Original data table (each row is an observation, each column is a variable).
        correlation_matrix (numpy.ndarray): Correlation matrix of the variables.

    Returns:
        numpy.ndarray: Matrix of principal components.
    """
    # Verify that the correlation matrix is square
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("The correlation matrix must be square.")

    # Verify that the number of variables matches the size of the correlation matrix
    if data.shape[1] != correlation_matrix.shape[0]:
        raise ValueError("The number of columns in the data must match the size of the correlation matrix.")

        # Calculate the mean and population standard deviation
    # Revisar si aquí se debería centrar con respecto a la media Riemanniana e igual la desviación estándar
    mean_centered_data = data - np.mean(data, axis=0)
    std_population = np.sqrt(np.sum(mean_centered_data ** 2, axis=0) / data.shape[0])

    # Standardize the data
    standardized_data = mean_centered_data / std_population

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the principal components
    principal_components = np.dot(standardized_data, eigenvectors)

    return principal_components

def correlation_variables_components(data, components):
    """
      Parameters:
    data (numpy array): The original data.
    components (numpy array): Components (2D array).

    Returns:
    correlations (pandas DataFrame): A DataFrame with two columns: the correlation of the first component with each variable in `data`,
                                     and the correlation of the second component with each variable in `data`.
    """
    # Combine original data and UMAP components into one DataFrame
    combined_data = pd.DataFrame(np.hstack((data, components[:, 0:2])),
                                 columns=[f'feature_{i + 1}' for i in range(data.shape[1])] + ['Component_1',
                                                                                               'Component_2'])

    # Calculate the Classic covariance matrix for the combined data
    # Revisar si aquí debería ser la matriz de covarianzas Riemanniana
    cov_matrix = covariance_matrix(combined_data)
    # print(cov_matrix)

    # Initialize a DataFrame to store the correlations
    correlations = pd.DataFrame(index=[f'feature_{i + 1}' for i in range(data.shape[1])],
                                columns=['Component_1', 'Component_2'])

    # Calculate the correlations for the first component
    for i in range(data.shape[1]):  # Loop through original data columns
        correlations.loc[f'feature_{i + 1}', 'Component_1'] = cov_matrix[i, -2] / np.sqrt(
            cov_matrix[i, i] * cov_matrix[-2, -2])

    # Calculate the correlations for the second component
    for i in range(data.shape[1]):  # Loop through original data columns
        correlations.loc[f'feature_{i + 1}', 'Component_2'] = cov_matrix[i, -1] / np.sqrt(
            cov_matrix[i, i] * cov_matrix[-1, -1])

    return correlations

def principal_components(data, corr_matrix):
    """
    Calcula los componentes principales a partir de la matriz de correlación (método clásico).
    """
    mean_centered_data = data - np.mean(data, axis=0)
    std_population = np.sqrt(np.sum(mean_centered_data ** 2, axis=0) / data.shape[0])
    standardized_data = mean_centered_data / std_population
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return np.dot(standardized_data, eigenvectors[:, sorted_indices])
