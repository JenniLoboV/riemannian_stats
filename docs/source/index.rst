RiemannStats
============

**RiemannStats: Statistical Analysis on Riemannian Manifolds**

**RiemannStats** is an open-source package that implements a novel principal component analysis methodology adapted for data on Riemannian manifolds, using UMAP as a core tool to construct the underlying geometric structure. This tool enables advanced statistical techniques to be applied to any type of dataset, honoring its local geometry, without requiring the data to originate from traditionally geometric domains like medical imaging or shape analysis.

Instead of assuming data resides in Euclidean space, RiemannStats transforms any data table into a Riemannian manifold by leveraging the local connectivity extracted from a UMAP-generated k-nearest neighbor graph. On top of this structure, the package computes Riemannian principal components, covariance and correlation matrices, and even provides 2D and 3D visualizations that faithfully capture the dataset’s topology.

With **RiemannStats**, you can:

* Incorporate the local geometry of your data for meaningful dimensionality reduction.
* Generate visual representations that better reflect the true structure of your data.
* Use a unified framework that generalizes classical statistical analysis to complex geometric contexts.
* Apply these techniques to both synthetic and real high-dimensional datasets.

This package is ideal for researchers, data scientists, and developers seeking to move beyond the traditional assumptions of classical statistics, applying models that respect the intrinsic structure of data.

---

.. toctree::
   :maxdepth: 1
   :caption: Menu

   How to Use RiemannStats        <examples>
   Install RiemannStats           <installation>
   Modules                        <menu>
   Scientific Paper               <paper>
