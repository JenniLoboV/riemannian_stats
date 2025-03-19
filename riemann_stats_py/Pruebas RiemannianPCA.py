import pandas as pd
from analysis import UMAPRiemannianAnalysis
from visualization import Visualization
from utilities import pca_inertia_by_components  # Función en estilo funcional

# ---------------------------
# Ejemplo 1: Datos Data10D_250
# ---------------------------
# Cargar el dataset Data10D_250
data_10d250 = pd.read_csv("../Archivos_Proyecto_PPS/Data10D_250.csv", sep=",", decimal=".")
n_neighbors_10d250 = 580

# Si existe la columna 'cluster', la usamos para identificar los grupos
if 'cluster' in data_10d250.columns:
    clusters_10d250 = data_10d250['cluster']
    clusters_10d250_3D = data_10d250
    data_10d250 = data_10d250.iloc[:, :-1]
else:
    clusters_10d250 = None

# Crear una instancia de UMAPRiemannianAnalysis para Data10D_250
analysis_10d250 = UMAPRiemannianAnalysis(data_10d250, n_neighbors=n_neighbors_10d250)

# Calcular el grafo de UMAP y la matriz rho para Data10D_250
umap_similarities_10d250 = analysis_10d250.calculate_umap_graph_similarities()
rho_10d250 = analysis_10d250.calculate_rho_matrix()

# Calcular las diferencias riemannianas y la matriz de distancias UMAP para Data10D_250
riemann_diff_10d250 = analysis_10d250.riemannian_vector_difference()
umap_distance_matrix_10d250 = analysis_10d250.calculate_umap_distance_matrix()

# Calcular la matriz de correlación riemanniana y extraer los componentes principales para Data10D_250
riem_cov_matrix_10d250 = analysis_10d250.riemannian_covariance_matrix()
riem_corr_10d250 = analysis_10d250.riemannian_correlation_matrix()
riemann_components_10d250 = analysis_10d250.riemannian_components_from_data_and_correlation(riem_corr_10d250)

# Calcular la inercia explicada (por ejemplo, comp1 = 0 y comp2 = 1)
comp1, comp2 = 0, 1
inertia_10d250 = pca_inertia_by_components(riem_corr_10d250, comp1, comp2) * 100

# Calcular las correlaciones entre las variables originales y los dos primeros componentes
correlations_10d250 = analysis_10d250.riemannian_correlation_variables_components(riemann_components_10d250)

# Visualizar el gráfico 3D de dispersión, usando el DataFrame original
Visualization.plot_3d_scatter_with_clusters(
    clusters_10d250_3D,  # Este DataFrame conserva la columna 'cluster'
    x_col='x',
    y_col='y',
    z_col='var1',
    cluster_col='cluster',
    title="3D Scatter Plot of Clusters",
    figsize=(12, 8)
)

# Visualizar el plano principal para Data10D_250, usando clusters si están definidos
if clusters_10d250 is not None:
    Visualization.plot_principal_plane_with_clusters(
        data_10d250,
        riemann_components_10d250,
        clusters_10d250,
        inertia_10d250,
        title="Principal Plane With Clusters - Data10D_250.csv"
    )
else:
    Visualization.plot_principal_plane(
        data_10d250,
        riemann_components_10d250,
        inertia_10d250,
        title="Plano Principal - Data10D_250.csv"
    )

# Visualizar el círculo de correlación para Data10D_250
Visualization.plot_correlation_circle(
    data_10d250,
    correlations_10d250,
    inertia_10d250,
    title="Correlation Circle - Data10D_250.csv"
)
