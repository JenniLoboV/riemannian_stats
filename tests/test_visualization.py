import unittest
import pandas as pd
from riemann_stats_py import Visualization, RiemannianUMAPAnalysis, pca_inertia_by_components

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # Set up a simple dataset for testing
        self.data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
            'cluster': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        })

        self.analysis = RiemannianUMAPAnalysis(self.data[['a', 'b']], n_neighbors=2)
        self.analysis.calculate_umap_graph_similarities()
        self.analysis.calculate_rho_matrix()
        self.analysis.riemannian_vector_difference()
        self.analysis.calculate_umap_distance_matrix()

        self.corr_matrix = self.analysis.riemannian_correlation_matrix()
        self.components = self.analysis.riemannian_components(self.corr_matrix)
        self.inertia = pca_inertia_by_components(self.corr_matrix, 0, 1) * 100

        self.viz = Visualization(data=self.data,
                                 components=self.components,
                                 explained_inertia=self.inertia,
                                 clusters=self.data['cluster'].values)

    def test_plot_principal_plane(self):
        # Test principal plane plotting without error
        try:
            self.viz.plot_principal_plane(title="Test Principal Plane")
        except Exception as e:
            self.fail(f"plot_principal_plane raised an exception unexpectedly: {e}")

    def test_plot_principal_plane_with_clusters(self):
        # Test principal plane plotting with clusters
        try:
            self.viz.plot_principal_plane_with_clusters(title="Test Principal Plane with Clusters")
        except Exception as e:
            self.fail(f"plot_principal_plane_with_clusters raised an exception unexpectedly: {e}")

    def test_plot_correlation_circle(self):
        correlations = self.analysis.riemannian_correlation_variables_components(self.components)
        try:
            self.viz.plot_correlation_circle(correlations=correlations, title="Test Correlation Circle")
        except Exception as e:
            self.fail(f"plot_correlation_circle raised an exception unexpectedly: {e}")

    def test_plot_2d_scatter_with_clusters(self):
        # Test 2D scatter plotting with clusters
        try:
            self.viz.plot_2d_scatter_with_clusters(x_col='a', y_col='b', cluster_col='cluster',
                                                   title="Test 2D Scatter with Clusters")
        except Exception as e:
            self.fail(f"plot_2d_scatter_with_clusters raised an exception unexpectedly: {e}")

    def test_plot_3d_scatter_with_clusters(self):
        # Extend dataset for a 3D scatter test
        self.data['c'] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.viz.data = self.data
        try:
            self.viz.plot_3d_scatter_with_clusters(x_col='a', y_col='b', z_col='c', cluster_col='cluster',
                                                   title="Test 3D Scatter with Clusters")
        except Exception as e:
            self.fail(f"plot_3d_scatter_with_clusters raised an exception unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()
