# -- Project Information --

project = 'RiemannStats'
author = 'Tu Nombre o Equipo'
release = '1.0.0'

# -- General Configuration --

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []

# -- HTML Output Options --

html_theme = 'furo'

html_title = "RiemannStats"
# html_favicon = "_static/images/favicon.ico"     # opcional

html_theme_options = {
    "light_logo": "images/light_logo.jpg",
    "dark_logo": "images/dark_logo.jpg",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}


html_static_path = ['_static']
html_css_files = [
    'css/custom.css',  # tu archivo de estilos personalizados
]

# -- Autodoc Settings --

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Napoleon Settings (for Google/NumPy docstrings) --

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Extension --

todo_include_todos = True
source_suffix = '.rst'
