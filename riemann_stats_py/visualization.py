from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Visualization:
    """
    Class for visualizing charts related to UMAP and PCA.

    Attributes:
        data (pandas.DataFrame): Data used for visualization.
        components (numpy.ndarray, optional): Matrix of principal components.
        explained_inertia (float): Explained inertia percentage.
        clusters (numpy.ndarray, optional): Array of cluster labels for each data point.
    """

    def __init__(self, data: pd.DataFrame, components: Optional[np.ndarray] = None,
                 explained_inertia: float = 0.0, clusters: Optional[np.ndarray] = None) -> None:
        """
        Initializes the Visualization object with the provided data, principal components,
        explained inertia, and clusters.

        Parameters:
            data (pandas.DataFrame): Data to be visualized.
            components (numpy.ndarray, optional): Principal components matrix. Defaults to None.
            explained_inertia (float, optional): Explained inertia percentage. Defaults to 0.0.
            clusters (numpy.ndarray, optional): Cluster labels. Defaults to None.
        """
        self.data = data
        self.components = components
        self.explained_inertia = explained_inertia
        self.clusters = clusters

    def plot_principal_plane(self, title: str = "") -> None:
        """
        Generates a plot of the principal plane using the principal components.

        Parameters:
            title (str, optional): Custom title to add above the default title.
        """
        default_title = "Principal Plane"
        if title:
            full_title = f"{title}\n{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        else:
            full_title = f"{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        # Assumes self.components is a numpy array with at least 2 columns.
        x, y = self.components[:, 0], self.components[:, 1]
        plt.scatter(x, y, color="gray")
        # Use the data index for labels.
        for i, label in enumerate(self.data.index):
            plt.text(x[i], y[i], label, fontsize=9, ha="right")
        plt.title(full_title)
        plt.axhline(y=0, color="dimgrey", linestyle="--")
        plt.axvline(x=0, color="dimgrey", linestyle="--")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

    def plot_principal_plane_with_clusters(self, title: str = "") -> None:
        """
        Generates a plot of the principal plane with points colored according to clusters.

        Parameters:
            title (str, optional): Custom title to add above the default title.

        Raises:
            ValueError: If cluster information is not provided.
        """
        if self.clusters is None:
            raise ValueError("Cluster information is required for this plot.")
        default_title = "Principal Plane With Clusters"
        if title:
            full_title = f"{title}\n{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        else:
            full_title = f"{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        x, y = self.components[:, 0], self.components[:, 1]
        plt.figure(figsize=(10, 8))
        unique_clusters = np.unique(self.clusters)
        for cluster in unique_clusters:
            cluster_points = self.clusters == cluster
            plt.scatter(x[cluster_points], y[cluster_points], label=f"Cluster {cluster}", alpha=0.7)
        for i, label in enumerate(self.data.index):
            plt.text(x[i], y[i], label, fontsize=8, ha="right")
        plt.title(full_title)
        plt.axhline(y=0, color="dimgrey", linestyle="--")
        plt.axvline(x=0, color="dimgrey", linestyle="--")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.show()

    def plot_correlation_circle(self, correlations: pd.DataFrame, title: str = "", scale: float = 1,
                                draw_circle: bool = True) -> None:
        """
        Generates a correlation circle for the principal components.

        Parameters:
            correlations (pandas.DataFrame): DataFrame containing the correlations for each variable.
            title (str, optional): Custom title to add above the default title.
            scale (float, optional): Scaling factor for the arrows. Defaults to 1.
            draw_circle (bool, optional): Whether to draw the unit circle. Defaults to True.
        """
        default_title = "Correlation Circle"
        if title:
            full_title = f"{title}\n{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        else:
            full_title = f"{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        if draw_circle:
            circle = plt.Circle((0, 0), radius=1.05, color="steelblue", fill=False)
            plt.gca().add_patch(circle)
        plt.axis("scaled")
        plt.axhline(y=0, color="dimgrey", linestyle="--")
        plt.axvline(x=0, color="dimgrey", linestyle="--")
        for i in range(correlations.shape[0]):
            plt.arrow(0, 0, correlations.iloc[i, 0] * scale, correlations.iloc[i, 1] * scale,
                      color="steelblue", alpha=0.5, head_width=0.05, head_length=0.05)
            plt.text(correlations.iloc[i, 0] * scale, correlations.iloc[i, 1] * scale,
                     self.data.columns[i], fontsize=9, ha="right")
        plt.title(full_title)
        plt.show()

    def plot_2d_scatter_with_clusters(self, x_col: str, y_col: str, cluster_col: str,
                                      title: str = "", figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Generates a 2D scatter plot colored by cluster.

        Parameters:
            x_col (str): Name of the column for the x-axis.
            y_col (str): Name of the column for the y-axis.
            cluster_col (str): Name of the column containing cluster labels.
            title (str, optional): Custom title to add above the default title.
            figsize (tuple, optional): Figure size. Defaults to (10, 8).
        """
        default_title = "2D Cluster Projection – Visualization of Groupings"
        if title:
            full_title = f"{title}\n{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        else:
            full_title = f"{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        plt.figure(figsize=figsize)
        unique_clusters = np.unique(self.data[cluster_col])
        for cluster in unique_clusters:
            subset = self.data[self.data[cluster_col] == cluster]
            plt.scatter(subset[x_col], subset[y_col], label=f"Cluster {cluster}", s=20, edgecolor="k")
        plt.title(full_title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.axis("equal")
        plt.legend(title="Clusters", loc="best", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    def plot_3d_scatter_with_clusters(self, x_col: str, y_col: str, z_col: str, cluster_col: str,
                                      title: str = "", figsize: Tuple[int, int] = (12, 8),
                                      cmap: str = "viridis", s: int = 50, alpha: float = 0.7) -> None:
        """
        Creates a 3D scatter plot colored by cluster.

        Parameters:
            x_col (str): Name of the column for the x-axis.
            y_col (str): Name of the column for the y-axis.
            z_col (str): Name of the column for the z-axis.
            cluster_col (str): Name of the column containing cluster labels.
            title (str, optional): Custom title to add above the default title.
            figsize (tuple, optional): Figure size. Defaults to (12, 8).
            cmap (str, optional): Colormap to use. Defaults to "viridis".
            s (int, optional): Size of the points. Defaults to 50.
            alpha (float, optional): Transparency of the points. Defaults to 0.7.
        """
        default_title = "3D Scatter Plot – Cluster Distribution"
        if title:
            full_title = f"{title}\n{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        else:
            full_title = f"{default_title} (Explained Inertia: {self.explained_inertia:.2f}%)"
        unique_clusters = np.unique(self.data[cluster_col])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(self.data[x_col], self.data[y_col], self.data[z_col],
                             c=self.data[cluster_col], cmap=cmap, s=s, alpha=alpha)
        for cluster in unique_clusters:
            color = plt.cm.get_cmap(cmap)(scatter.norm(cluster))
            ax.scatter([], [], [], color=color, label=f"Cluster {cluster}")
        ax.set_title(full_title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.legend(title="Clusters", loc="upper left", bbox_to_anchor=(1, 0.8))
        plt.tight_layout()
        plt.show()
