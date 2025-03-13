import numpy as np

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
