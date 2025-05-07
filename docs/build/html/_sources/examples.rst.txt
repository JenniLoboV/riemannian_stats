How to Use RiemannStats
========================

RiemannStats provides an intuitive and powerful interface to perform Principal Component Analysis on Riemannian manifolds derived from any tabular dataset. Below you’ll find two comprehensive examples that demonstrate how to load data, configure the analysis, compute Riemannian structures, and visualize the results.

Each example below illustrates how to apply this process to different types of datasets: one classical and low-dimensional, the other high-dimensional and synthetic.

Example 1: Iris Dataset
-----------------------

This example uses the classic Iris dataset to showcase RiemannStats' ability to uncover structure in well-known data. It demonstrates:

- Loading and preprocessing the data using DataProcessing.load_data.
- Detecting clusters (from the 'tipo' column).
- Applying Riemannian UMAP analysis to compute graph similarities, vector differences, distances, and correlation matrices.
- Extracting principal components and calculating explained inertia.
- Visualizing results using 2D and 3D scatter plots, principal planes, and correlation circles.

This example is ideal for those looking to understand the workflow and inspect meaningful geometry in low-dimensional datasets

**Insertar html generado por rstudio**

.. literalinclude:: ../../examples/example1.py
   :language: python
   :caption: Example 1

Example 2: Data10D_250
----------------------

This example applies RiemannStats to a synthetic, high-dimensional dataset with known cluster structure. It demonstrates:

- Handling datasets with more variables and complex geometry.
- Separating clustering labels (from the `'cluster'` column) from the analysis data.
- Running a full Riemannian PCA pipeline adapted for higher dimensions.
- Generating detailed visualizations that reveal cluster relationships in 2D and 3D spaces.
- Showing how R-PCA preserves and enhances the interpretability of complex datasets compared to classical PCA.

This example highlights the robustness of RiemannStats when dealing with real-world scenarios where dimensionality reduction and structure preservation are critical.

**Insertar html generado por rstudio**


.. literalinclude:: ../../examples/example2.py
   :language: python
   :caption: Example 2
