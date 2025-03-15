import pandas as pd
from analysis import UMAPRiemannianAnalysis
from visualization import Visualization
from utilities import pca_inertia_by_components  # Función en estilo funcional

# ---------------------------
# Ejemplo 1: Datos Iris
# ---------------------------
# Cargar el dataset Iris
data_iris = pd.read_csv("../Archivos_Proyecto_PPS/iris.csv", sep=";", decimal=".")
# Se asume que la última columna es la etiqueta de cluster (por ejemplo, 'tipo') y queremos trabajar solo con las features:
data_iris_features = data_iris.iloc[:, :-1]

# Crear una instancia de UMAPRiemannianAnalysis para Iris (por ejemplo, usando 50 vecinos)
analysis_iris = UMAPRiemannianAnalysis(data_iris_features, n_neighbors=50, min_dist=0.1, metric='euclidean')

# Calcular el grafo de UMAP y la matriz rho para Iris
umap_similarities_iris = analysis_iris.calculate_umap_graph_similarities()
rho_iris = analysis_iris.calculate_rho_matrix()

# Calcular las diferencias riemannianas y la matriz de distancias UMAP para Iris
riemann_diff_iris = analysis_iris.riemannian_vector_difference()
umap_distance_matrix_iris = analysis_iris.calculate_umap_distance_matrix()

# Calcular la matriz de correlación riemanniana y extraer los componentes principales para Iris
riemann_corr_iris = analysis_iris.riemannian_correlation_matrix()
riemann_components_iris = analysis_iris.riemannian_components(riemann_corr_iris)

# Calcular la inercia explicada por los dos primeros componentes (en porcentaje)
inertia_iris = pca_inertia_by_components(riemann_corr_iris, 0, 1) * 100

# Visualizar el plano principal Riemanniano para Iris
Visualization.plot_principal_plane(data_iris_features, riemann_components_iris, inertia_iris, title="Plano Principal Riemanniano - Iris")


# ---------------------------
# Ejemplo 2: Datos Data10D_250
# ---------------------------
# Cargar el dataset Data10D_250
data_10d250 = pd.read_csv("../Archivos_Proyecto_PPS/Data10D_250.csv", sep=",", decimal=".")
n_neighbors_10d250 = 580

# Si existe la columna 'cluster', la usamos para identificar los grupos
if 'cluster' in data_10d250.columns:
    clusters_10d250 = data_10d250['cluster']
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
