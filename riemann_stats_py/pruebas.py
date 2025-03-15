import pandas as pd
from RiemannianPCA import UMAPRiemannianAnalysis
from visualization import Visualization
from ClassicPCA import pca_inertia_by_components  # Si lo tienes en estilo funcional

# ---------------------------
# Ejemplo 1: Datos Iris
# ---------------------------
# Cargar los datos (en este ejemplo usamos el dataset Iris)
data = pd.read_csv("iris.csv", sep=";", decimal=".")
# Suponemos que la última columna es la etiqueta de cluster y queremos trabajar solo con las features
data_iris = data.iloc[:, :-1]

# Crear una instancia de UMAPAnalysis con un número de vecinos adecuado (por ejemplo, 50)
analysis = UMAPRiemannianAnalysis(data_iris, n_neighbors=50, min_dist=0.1, metric='euclidean')

# Calcular el grafo de UMAP y la matriz rho
umap_similarities = analysis.calculate_umap_graph_similarities()
rho = analysis.calculate_rho_matrix()

# Calcular las diferencias riemannianas y la matriz de distancias UMAP
riemann_diff = analysis.riemannian_vector_difference()
umap_distance_matrix = analysis.calculate_umap_distance_matrix()

# Calcular la matriz de correlación riemanniana y extraer los componentes principales riemannianos
riemann_corr = analysis.riemannian_correlation_matrix()
riemann_components = analysis.riemannian_components(riemann_corr)

# Calcular la inercia explicada por los dos primeros componentes (usando la función funcional)
inertia = pca_inertia_by_components(data_iris, riemann_corr, 0, 1) * 100

# Visualizar el plano principal
Visualization.plot_principal_plane(data_iris, riemann_components, inertia, title="Plano Principal Riemanniano")

# ---------------------------
# Ejemplo 2: Datos EjemploEstudiantes
# ---------------------------
# Cargar los datos de EjemploEstudiantes
data_estudiantes = pd.read_csv("EjemploEstudiantes.csv", sep=";", decimal=",", index_col=0)
cantidad_vecinos_estudiantes = 3

# Si existe la columna 'grupo', la usamos para identificar clusters
if 'grupo' in data_estudiantes.columns:
    cl_estudiantes = data_estudiantes['grupo']
else:
    cl_estudiantes = None  # O asignar otro identificador

# Crear una instancia de UMAPAnalysis usando los parámetros definidos
analysis_est = UMAPRiemannianAnalysis(data_estudiantes, n_neighbors=cantidad_vecinos_estudiantes, min_dist=0.1, metric='euclidean')

# Calcular el grafo de UMAP y la matriz Rho
umap_simil_est = analysis_est.calculate_umap_graph_similarities()
p_rho_est = analysis_est.calculate_rho_matrix()

# Calcular las diferencias Riemannianas y la matriz de distancias UMAP
riem_diff_est = analysis_est.riemannian_vector_difference()
umap_dist_est = analysis_est.calculate_umap_distance_matrix()

# Calcular la matriz de correlación Riemanniana y extraer los componentes principales
riem_cor_est = analysis_est.riemannian_correlation_matrix()
riem_comp_est = analysis_est.riemannian_components(riem_cor_est)

# Calcular la inercia explicada por los dos primeros componentes (por ejemplo, usando comp1=0 y comp2=1)
comp1, comp2 = 0, 1
inercia_est = pca_inertia_by_components(data_estudiantes, riem_cor_est, comp1, comp2) * 100

# Visualizar el plano principal, usando clusters si se definieron
if cl_estudiantes is not None:
    Visualization.plot_principal_plane_with_clusters(
        data_estudiantes,
        riem_comp_est,
        cl_estudiantes,
        inercia_est,
        title="Plano Principal EjemploEstudiantes"
    )
else:
    Visualization.plot_principal_plane(
        data_estudiantes,
        riem_comp_est,
        inercia_est,
        title="Plano Principal EjemploEstudiantes"
    )
