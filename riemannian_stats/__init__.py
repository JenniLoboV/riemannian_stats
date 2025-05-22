"""
riemannian_stats

This package provides tools for Riemannian statistical analysis using UMAP, including:
- Data preprocessing
- Local distance computations
- Riemannian PCA
- Interactive visualizations
"""

# Import with original class names (PascalCase)
from .data_processing import DataProcessing
from .riemannian_umap_analysis import RiemannianUMAPAnalysis
from .visualization import Visualization
from .utilities import Utilities

# Also provide lowercase aliases for user-friendly imports
from .data_processing import DataProcessing as dataprocessing
from .riemannian_umap_analysis import RiemannianUMAPAnalysis as riemannianumapanalysis
from .visualization import Visualization as visualization
from .utilities import Utilities as utilities

__all__ = [
    # PascalCase
    "DataProcessing",
    "RiemannianUMAPAnalysis",
    "Visualization",
    "Utilities",

    # lowercase aliases
    "dataprocessing",
    "riemannianumapanalysis",
    "visualization",
    "utilities"
]
