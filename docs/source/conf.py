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

html_title = "RiemannStats Documentation"
html_logo = "_static/images/logo-light.png"     # si tienes un logo
html_favicon = "_static/images/favicon.ico"     # opcional

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_logo": "logo-light.jng",      # asegúrate de tener estos archivos
    "dark_logo": "logo-dark.jng",
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
