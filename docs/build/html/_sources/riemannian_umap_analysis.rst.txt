RiemannianUMAPAnalysis
======================

The ``RiemannianUMAPAnalysis`` module offers a sophisticated approach for dimensionality reduction and data analysis by combining Uniform Manifold Approximation and Projection (UMAP) with Riemannian geometry. This module enables users to explore and analyze high-dimensional datasets through the lens of Riemannian geometry, providing advanced tools for extracting meaningful patterns and structures from data.

Overview
--------

This class provides methods for UMAP-based dimensionality reduction, extended by incorporating Riemannian metrics to better capture geometric properties of the data. With the ability to analyze data using various distance metrics, ``RiemannianUMAPAnalysis`` allows for flexibility in working with different types of datasets, including those that require non-Euclidean geometry.

Features:
---------
- **UMAP-based Analysis**: Leverages UMAP for non-linear dimensionality reduction, a powerful tool for visualizing and understanding high-dimensional data.
- **Riemannian Geometry**: Incorporates Riemannian principles to adjust the analysis based on the intrinsic geometry of the data, offering a more meaningful interpretation of distances and similarities in some applications.
- **Correlation Analysis**: Includes methods for calculating Riemannian correlation matrices and performing principal component analysis (PCA) using Riemannian techniques.
- **Distance Metrics Flexibility**: Supports custom distance metrics for UMAP, allowing for a tailored approach to different datasets.


When to Use:
------------
This module is particularly useful for:

* High-dimensional data where traditional dimensionality reduction methods fall short.
* Applications that require advanced distance metrics or non-Euclidean spaces.
* Users looking to uncover deep relationships in data using Riemannian geometry principles.

By utilizing this module, you can significantly enhance your data analysis workflow, gaining insights that may be hidden when using more traditional approaches.

Module Documentation
---------------------
For detailed information on the `RiemannianUMAPAnalysis` module, refer to the module's API documentation:

.. automodule:: riemannian_stats.riemannian_umap_analysis
   :members:
   :show-inheritance:
   :undoc-members:
