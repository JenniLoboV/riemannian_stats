import pandas as pd
from riemannian_umap_analysis import RiemannianUMAPAnalysis
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
    # Conservamos una copia del DataFrame original con la columna 'cluster' para visualizaciones 3D y 2D
    clusters_10d250_3D = data_10d250.copy()
    # Removemos la columna 'cluster' para el análisis (si es necesario)
    data_10d250 = data_10d250.iloc[:, :-1]
else:
    clusters_10d250 = None
    clusters_10d250_3D = data_10d250

# Crear una instancia de UMAPRiemannianAnalysis para Data10D_250
analysis_10d250 = RiemannianUMAPAnalysis(data_10d250, n_neighbors=n_neighbors_10d250)

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

# Crear una instancia de Visualization. Si se cuenta con clusters, se pasan; de lo contrario, se omiten.
if clusters_10d250 is not None:
    viz = Visualization(data=clusters_10d250_3D,
                        components=riemann_components_10d250,
                        explained_inertia=inertia_10d250,
                        clusters=clusters_10d250)
else:
    viz = Visualization(data=data_10d250,
                        components=riemann_components_10d250,
                        explained_inertia=inertia_10d250)

# Visualizar el gráfico 2D de dispersión (se usa la columna 'cluster' del DataFrame)
viz.plot_2d_scatter_with_clusters(x_col="x", y_col="y", cluster_col="cluster", title="Data10D_250.csv")

# Visualizar el gráfico 3D de dispersión (se usa la columna 'cluster' del DataFrame)
viz.plot_3d_scatter_with_clusters(x_col="x", y_col="y", z_col="var1", cluster_col="cluster",
                                  title="Data10D_250.csv", figsize=(12, 8))

# Visualizar el plano principal para Data10D_250, usando clusters si están definidos
if clusters_10d250 is not None:
    viz.plot_principal_plane_with_clusters(title="Data10D_250.csv")
else:
    viz.plot_principal_plane(title="Data10D_250.csv")

# Visualizar el círculo de correlación para Data10D_250
viz.plot_correlation_circle(correlations=correlations_10d250, title="Data10D_250.csv")
