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
    def plot_principal_plane_with_clusters(data, components, clusters, explained_inertia, title="Principal Plane With Clusters"):
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
    def plot_correlation_circle(data, correlations, explained_inertia, title="Correlation Circle", scale=1, draw_circle=True):
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
            plt.arrow(0, 0, correlations.iloc[i, 0] * scale, correlations.iloc[i, 1] * scale, color='steelblue', alpha=0.5, head_width=0.05, head_length=0.05)
            plt.text(correlations.iloc[i, 0] * scale, correlations.iloc[i, 1] * scale, data.columns[i], fontsize=9, ha='right')
        plt.title(f"{title} (Explained Inertia: {explained_inertia:.2f}%)")
        plt.show()
