How to Use RiemannStats
========================

RiemannStats provides an intuitive and powerful interface to perform Principal Component Analysis on Riemannian manifolds derived from any tabular dataset. Below you’ll find two comprehensive examples that demonstrate how to load data, configure the analysis, compute Riemannian structures, and visualize the results.

Each example below illustrates how to apply this process to different types of datasets: one classical and low-dimensional, the other high-dimensional and synthetic.

Example 1: Iris Dataset
-----------------------

Este ejemplo utiliza el famoso conjunto de datos Iris para mostrar la capacidad de RiemannStats de revelar estructura en datos bien conocidos. Demuestra:

- Carga y preprocesamiento de datos con `DataProcessing.load_data`.
- Detección de clústeres a partir de la columna `'tipo'`.
- Aplicación del análisis Riemanniano con UMAP para calcular similitudes del grafo, diferencias vectoriales, distancias y matrices de correlación.
- Extracción de componentes principales e inercia explicada.
- Visualización de resultados mediante gráficos 2D, 3D, planos principales y círculos de correlación.

Es ideal para comprender el flujo de trabajo y explorar estructuras significativas en un conjunto de datos de baja dimensión.

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

.. literalinclude:: ../../examples/example2.py
   :language: python
   :caption: Example 2
