from riemann_stats_py import RiemannianUMAPAnalysis, Visualization, DataProcessing, pca_inertia_by_components

# ---------------------------
# Example 1: Data10D_250 Dataset
# ---------------------------
# Load the Data10D_250.csv dataset using DataProcessing.load_data, specifying the separator and decimal format.
data = DataProcessing.load_data("./data/Data10D_250.csv", separator=",", decimal=".")

# Define the number of neighbors as the length of the data divided by the number of clusters (in this example, 5).
n_neighbors = int(len(data) / 5)

# Check if the 'clusters' column exists to identify groups (clusters).
if 'cluster' in data.columns:
    clusters = data['cluster']
    # Keep a copy of the original DataFrame (with the 'cluster' column) for 3D and 2D visualizations
    data_with_clusters = data.copy()
    # Remove the 'cluster' column for analysis (if needed)
    data = data.iloc[:, :-1]
else:
    clusters = None
    data_with_clusters = data

# Create an instance of RiemannianUMAPAnalysis for the dataset
analysis = RiemannianUMAPAnalysis(data, n_neighbors=n_neighbors)

# --------------------------------------------------------
# Compute UMAP graph similarities and rho matrix for the data
# --------------------------------------------------------
umap_similarities = analysis.calculate_umap_graph_similarities()
print("calculate_umap_graph_similarities:", umap_similarities)

rho = analysis.calculate_rho_matrix()
print("calculate_rho_matrix:", rho)

# --------------------------------------------------------
# Compute Riemannian vector differences and UMAP distance matrix
# --------------------------------------------------------
riemannian_diff = analysis.riemannian_vector_difference()
print("riemannian_vector_difference:", riemannian_diff)

umap_distance_matrix = analysis.calculate_umap_distance_matrix()
print("calculate_umap_distance_matrix:", umap_distance_matrix)

# --------------------------------------------------------
# Compute the Riemannian covariance and correlation matrices, and extract principal components
# --------------------------------------------------------
riemann_cov_matrix = analysis.riemannian_covariance_matrix()
print("riemannian_covariance_matrix:", riemann_cov_matrix)

riemann_corr = analysis.riemannian_correlation_matrix()
print("riemannian_correlation_matrix:", riemann_corr)

riemann_components = analysis.riemannian_components_from_data_and_correlation(riemann_corr)
print("riemannian_components_from_data_and_correlation:", riemann_components)

# --------------------------------------------------------
# Compute the explained inertia (using components 0 and 1)
# --------------------------------------------------------
comp1, comp2 = 0, 1
inertia = pca_inertia_by_components(riemann_corr, comp1, comp2) * 100
print("pca_inertia_by_components:", inertia)

# --------------------------------------------------------
# Compute correlations between original variables and the first two principal components
# --------------------------------------------------------
correlations = analysis.riemannian_correlation_variables_components(riemann_components)
print("riemannian_correlation_variables_components:", correlations)

# --------------------------------------------------------
# Visualization: Create plots based on the availability of clusters.
# If clusters are provided, use cluster-based plots; otherwise, use plots without clusters.
# --------------------------------------------------------
if clusters is not None:
    # Create a Visualization instance including clusters
    viz = Visualization(data=data_with_clusters,
                        components=riemann_components,
                        explained_inertia=inertia,
                        clusters=clusters)
    try:
        # 1. 2D scatter plot with clusters (requires 'cluster' column in the DataFrame)
        viz.plot_2d_scatter_with_clusters(x_col="x", y_col="y", cluster_col="cluster", title="Data10D_250.csv")
    except Exception as e:
        print("2D scatter plot with clusters failed:", e)

    try:
        # 2. Principal plane with clusters
        viz.plot_principal_plane_with_clusters(title="Data10D_250.csv")
    except Exception as e:
        print("Principal plane with clusters plot failed:", e)

    try:
        # 3. 3D scatter plot with clusters (requires 'cluster' column and proper 3D data columns)
        viz.plot_3d_scatter_with_clusters(x_col="x", y_col="y", z_col="var1", cluster_col="cluster",
                                          title="Data10D_250.csv", figsize=(12, 8))
    except Exception as e:
        print("3D scatter plot with clusters failed:", e)
else:
    # Create a Visualization instance without clusters
    viz = Visualization(data=data,
                        components=riemann_components,
                        explained_inertia=inertia)
    try:
        # Plot principal plane (does not require cluster information)
        viz.plot_principal_plane(title="Data10D_250.csv")
    except Exception as e:
        print("Principal plane plot failed:", e)

# 4. Plot the correlation circle (should work regardless of clusters)
try:
    viz.plot_correlation_circle(correlations=correlations, title="Data10D_250.csv")
except Exception as e:
    print("Correlation circle plot failed:", e)
