"""
This script demonstrates a complete workflow for analyzing a small custom dataset using the riemann_stats_py package.

The dataset is defined as a DataFrame with:
- Two features ('a' and 'b')
- A 'cluster' column indicating group membership

The workflow is as follows:
1. The dataset is created and the 'cluster' column is separated from the data used for analysis.
2. An instance of RiemannianUMAPAnalysis is created with the analysis data (excluding clusters).
3. The script computes:
   - UMAP graph similarities and the corresponding Rho matrix,
   - Riemannian vector differences and the resulting UMAP distance matrix,
   - Riemannian covariance and correlation matrices,
   - Principal components derived from the correlation matrix.
4. It calculates the explained inertia (using the first two principal components) and then computes the correlations
   between the original variables and these principal components.
5. Finally, it uses the Visualization class to generate:
   - A 2D scatter plot with clusters,
   - A principal plane plot,
   - A 3D scatter plot (if applicable),
   - And a correlation circle plot.

This example illustrates how riemann_stats_py enables thorough analysis and visualization even on a small dataset.
"""

from riemann_stats_py import RiemannianUMAPAnalysis, Visualization, Utilities
import pandas as pd

# Create a small custom dataset.
# Note: The 'cluster' column must have the same number of rows as the other columns.
data = pd.DataFrame({
    'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'b': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
})

# If 'cluster' is present, extract it for visualization and remove it from the data used for analysis.
if 'cluster' in data.columns:
    clusters = data['cluster']
    data_with_clusters = data.copy()
    # Remove the 'cluster' column for analysis
    analysis_data = data.drop(columns=['cluster'])
else:
    clusters = None
    data_with_clusters = data
    analysis_data = data

# Instantiate the analysis class.
# We use n_neighbors=2 to keep the example simple.
analysis = RiemannianUMAPAnalysis(analysis_data, n_neighbors=2)

# --------------------------------------------------------
# Compute UMAP graph similarities and the rho matrix.
# --------------------------------------------------------
umap_similarities = analysis.calculate_umap_graph_similarities()
print("UMAP graph similarities:")
print(umap_similarities)

rho_matrix = analysis.calculate_rho_matrix()
print("Rho matrix:")
print(rho_matrix)

# --------------------------------------------------------
# Compute Riemannian vector differences and the UMAP distance matrix.
# --------------------------------------------------------
riemann_diff = analysis.riemannian_vector_difference()
print("Riemannian vector differences:")
print(riemann_diff)

umap_distance_matrix = analysis.calculate_umap_distance_matrix()
print("UMAP distance matrix:")
print(umap_distance_matrix)

# --------------------------------------------------------
# Compute the Riemannian covariance and correlation matrices,
# and extract principal components.
# --------------------------------------------------------
cov_matrix = analysis.riemannian_covariance_matrix()
print("Riemannian covariance matrix:")
print(cov_matrix)

corr_matrix = analysis.riemannian_correlation_matrix()
print("Riemannian correlation matrix:")
print(corr_matrix)

components = analysis.riemannian_components_from_data_and_correlation(corr_matrix)
print("Principal components (from data and correlation):")
print(components)

# --------------------------------------------------------
# Compute the explained inertia (using components 0 and 1).
# --------------------------------------------------------
comp1, comp2 = 0, 1
inertia = Utilities.pca_inertia_by_components(corr_matrix, comp1, comp2) * 100
print("Explained inertia (%):")
print(inertia)

# --------------------------------------------------------
# Compute correlations between original variables and the first two principal components.
# --------------------------------------------------------
correlations = analysis.riemannian_correlation_variables_components(components)
print("Correlation between original variables and principal components:")
print(correlations)

# --------------------------------------------------------
# Visualization: Create plots based on the availability of clusters.
# --------------------------------------------------------
if clusters is not None:
    # Create a Visualization instance including clusters.
    viz = Visualization(data=data_with_clusters,
                        components=components,
                        explained_inertia=inertia,
                        clusters=clusters)
    try:
        # 1. 2D scatter plot with clusters.
        viz.plot_2d_scatter_with_clusters(x_col="a", y_col="b", cluster_col="cluster", title="Small Dataset Example")
    except Exception as e:
        print("2D scatter plot with clusters failed:", e)

    try:
        # 2. Principal plane plot with clusters.
        viz.plot_principal_plane_with_clusters(title="Small Dataset Principal Plane")
    except Exception as e:
        print("Principal plane with clusters plot failed:", e)

    try:
        # 3. 3D scatter plot with clusters (using column 'a' as a dummy for a third dimension).
        viz.plot_3d_scatter_with_clusters(x_col="a", y_col="b", z_col="a", cluster_col="cluster",
                                          title="Small Dataset 3D Scatter", figsize=(8, 6))
    except Exception as e:
        print("3D scatter plot with clusters failed:", e)
else:
    viz = Visualization(data=analysis_data,
                        components=components,
                        explained_inertia=inertia)
    try:
        # Plot principal plane (does not require cluster information).
        viz.plot_principal_plane(title="Small Dataset Principal Plane")
    except Exception as e:
        print("Principal plane plot failed:", e)

# 4. Plot the correlation circle (should work regardless of clusters).
try:
    viz.plot_correlation_circle(correlations=correlations, title="Small Dataset Correlation Circle")
except Exception as e:
    print("Correlation circle plot failed:", e)
