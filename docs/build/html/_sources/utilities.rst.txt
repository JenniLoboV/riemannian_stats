Utilities
=========

The ``Utilities`` module provides a collection of common utility functions for data science projects. It includes static methods that facilitate mathematical and statistical operations, designed to simplify integration into larger data processing and analysis workflows. These methods can be used directly without the need for class instantiation.

Overview
--------

This class includes utility functions that can be used to perform frequently needed operations in data analysis tasks. The methods focus on providing easy access to common mathematical computations, making them convenient for use across different data science projects. By not requiring instantiation, the tools can be quickly integrated into any analysis pipeline.

Features
--------
- **PCA Inertia Calculation**: Calculates the inertia explained by two specific principal components in a Principal Component Analysis (PCA) task.
- **Statistical Tools**: Provides essential functions for analyzing data in a data science pipeline.

Methods
-------
- ``pca_inertia_by_components()``: This method calculates the inertia (explained variance) associated with two selected principal components of a correlation matrix, making it useful for understanding the significance of specific components in a PCA.

Example Usage
-------------
Here’s an example of how to use ``Utilities``:

```python
from riemann_stats_py import Utilities
import numpy as np

# Example correlation matrix
correlation_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])

# Calculate the inertia explained by the first two principal components
inertia = Utilities.pca_inertia_by_components(correlation_matrix, component1=0, component2=1)

print(f"Inertia explained by the first two components: {inertia}")
````

## When to Use

This module is particularly useful in data science workflows where common mathematical operations, such as PCA analysis, need to be performed. The function is designed to help quantify the importance of specific components in dimensionality reduction tasks, and its utility can be extended to a wide range of data analysis projects.

By using these utility functions, you can streamline your data analysis processes and avoid reinventing the wheel when performing standard mathematical operations.

Module Documentation
---------------------

For detailed information on the `Utilities` module, refer to the module's API documentation:

.. automodule:: riemann_stats_py.utilities
   :members:
   :show-inheritance:
   :undoc-members:


