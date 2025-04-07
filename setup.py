import platform
from setuptools import setup, find_packages

def readme():
    try:
        with open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        # Para versiones de Python que no soporten el parámetro encoding en open
        import io
        with io.open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()

configuration = {
    "name": "riemann_stats_py",
    "version": "0.1",
    "description": "Principal component analysis on Riemannian data.",
    "long_description": readme(),
    "long_description_content_type": "text/x-rst",
    "author": "Oldemar Rodríguez Rojas, Jennifer Lobo Vásquez",
    "author_email": "<oldemar.rodriguez@ucr.ac.cr, jennifer.lobo.vasquez@est.una.ac.cr>",
    "url": "http://github.com/tu_usuario/riemann_stats_py",  # Actualiza con la URL de tu repositorio
    "packages": find_packages(),
    "install_requires": [
        "matplotlib~=3.10.1",
        "pandas~=2.2.3",
        "numpy~=2.1.3",
        "scikit-learn~=1.6.1",
        "umap-learn"
    ],
    "python_requires": ">=3.6",
    "keywords": [
        "Riemannian Manifold",
        "Riemannian Principal Component Analysis (R-PCA)",
        "Riemannian Statistics",
        "Local Distance Notion",
        "Dimension Reduction",
        "Geometric Structures"
    ],
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    "license": "BSD-3-Clause",
    "extras_require": {
        "plot": [
            "matplotlib",
            "pandas"
        ]
    },
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "zip_safe": False,
}

setup(**configuration)
