import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'riemann_stats_py'
copyright = '2025, Oldemar Rodríguez Rojas, Jennifer Lobo Vásquez'
author = 'Oldemar Rodríguez Rojas, Jennifer Lobo Vásquez'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = []
