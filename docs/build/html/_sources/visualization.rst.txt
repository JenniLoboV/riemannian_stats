Visualization
=============

The ``Visualization`` module provides functionality for visualizing charts related to UMAP and PCA. This class includes various plotting methods that allow for an easy exploration of the data's principal components, clustering information, and correlations. It supports 2D and 3D scatter plots, as well as specialized visualizations like correlation circles and principal planes.

Overview
--------

This class is designed to facilitate data visualization, particularly for dimensionality reduction techniques like UMAP and PCA. It allows users to visualize the relationships between data points in a reduced-dimensional space, both with and without clustering information. The methods in this module generate interactive and static plots using `matplotlib` to support visual analysis.

Features
--------
- **Principal Plane Plotting**: Generate 2D scatter plots of the first two principal components, with options for custom titles and displaying inertia.
- **Clustered Principal Plane Plot**: Visualize the principal plane with points colored according to clusters, enabling the exploration of cluster structure in reduced space.
- **Correlation Circle Plot**: Create a correlation circle to visualize the relationships between original variables and principal components.
- **2D and 3D Scatter Plots**: Generate scatter plots for 2D and 3D projections with points colored by cluster membership, useful for visualizing high-dimensional data groupings.

When to Use
-----------

This module is especially useful when you want to visualize the results of dimensionality reduction techniques such as PCA or UMAP. It helps in understanding the structure of data, especially in high-dimensional spaces, and can be used to explore the relationship between different clusters in the data. The methods are designed for ease of use and can be customized to fit the needs of different types of data analysis tasks.

By using this module, you can enhance your data analysis pipeline with clear, informative visualizations that aid in interpreting complex datasets.


Module Documentation
---------------------
For detailed information on the `Visualization` module, refer to the module's API documentation:

.. automodule:: riemann_stats_py.visualization
   :members:
   :show-inheritance:
   :undoc-members: