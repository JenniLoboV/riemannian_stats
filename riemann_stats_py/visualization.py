import matplotlib.pyplot as plt
import numpy as np
class Visualization:
    """
    Clase para la visualización de gráficos relacionados con UMAP y PCA.
    """
    @staticmethod
    def plot_principal_plane(data, components, explained_inertia, title="Principal Plane"):
        """
        Genera un gráfico del plano principal con los componentes principales.
        """
        x, y = components[:, 0], components[:, 1]
        plt.scatter(x, y, color='gray')
        for i, label in enumerate(data.index):
            plt.text(x[i], y[i], label, fontsize=9, ha='right')
        plt.title(f"{title} (Explained Inertia: {explained_inertia:.2f}%)")
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()

    @staticmethod
    def plot_principal_plane_with_clusters(data, components, clusters, explained_inertia,
                                           title="Principal Plane With Clusters"):
        """
        Genera un gráfico del plano principal coloreando los puntos según los clústeres.
        """
        x, y = components[:, 0], components[:, 1]
        plt.figure(figsize=(10, 8))
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            cluster_points = (clusters == cluster)
            plt.scatter(x[cluster_points], y[cluster_points], label=f'Cluster {cluster}', alpha=0.7)
        for i, label in enumerate(data.index):
            plt.text(x[i], y[i], label, fontsize=8, ha='right')
        plt.title(f"{title} (Explained Inertia: {explained_inertia:.2f}%)")
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_correlation_circle(data, correlations, explained_inertia, title="Correlation Circle", scale=1,
                                draw_circle=True):
        """
        Genera un círculo de correlación para los componentes principales.
        """
        if draw_circle:
            circle = plt.Circle((0, 0), radius=1.05, color='steelblue', fill=False)
            plt.gca().add_patch(circle)
        plt.axis('scaled')
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        for i in range(correlations.shape[0]):
            plt.arrow(0, 0, correlations.iloc[i, 0] * scale, correlations.iloc[i, 1] * scale, color='steelblue',
                      alpha=0.5, head_width=0.05, head_length=0.05)
            plt.text(correlations.iloc[i, 0] * scale, correlations.iloc[i, 1] * scale, data.columns[i], fontsize=9,
                     ha='right')
        plt.title(f"{title} (Explained Inertia: {explained_inertia:.2f}%)")
        plt.show()

    @staticmethod
    def plot_3d_scatter_with_clusters(data, x_col, y_col, z_col, cluster_col, title="3D Scatter Plot of Clusters",
                                      figsize=(12, 8), cmap='viridis', s=50, alpha=0.7):
        """
        Crea un gráfico 3D de dispersión coloreado según la columna de clusters.

        Parámetros:
          - data: DataFrame que contiene los datos.
          - x_col, y_col, z_col: Nombres de las columnas que contienen las coordenadas X, Y y Z.
          - cluster_col: Nombre de la columna que contiene los clusters.
          - title: Título del gráfico (por defecto "3D Scatter Plot of Clusters").
          - figsize: Tamaño de la figura (por defecto (12,8)).
          - cmap: Colormap a utilizar (por defecto 'viridis').
          - s: Tamaño de los puntos (por defecto 50).
          - alpha: Transparencia de los puntos (por defecto 0.7).

        La función crea la leyenda asociada a cada cluster basado en los colores asignados.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Aunque no es estrictamente necesario en versiones modernas

        # Obtener clusters únicos
        unique_clusters = np.unique(data[cluster_col])

        # Crear figura y eje 3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Graficar los puntos, asignando colores según la columna de clusters
        scatter = ax.scatter(data[x_col], data[y_col], data[z_col],
                             c=data[cluster_col], cmap=cmap, s=s, alpha=alpha)

        # Crear leyenda: para cada cluster, se crea un "dummy" scatter con el color correspondiente
        for cluster in unique_clusters:
            # Obtener el color para este cluster a partir del colormap
            color = plt.cm.get_cmap(cmap)(scatter.norm(cluster))
            ax.scatter([], [], [], color=color, label=f'Cluster {cluster}')

        # Configurar títulos y etiquetas de ejes
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)

        ax.legend(title="Clusters", loc="upper left", bbox_to_anchor=(1, 0.8))
        plt.tight_layout()
        plt.show()
