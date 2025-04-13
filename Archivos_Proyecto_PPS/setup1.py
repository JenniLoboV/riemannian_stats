import platform
from setuptools import setup, find_packages
import io


def readme():
    try:
        with open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        with io.open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()

setup(
    name="riemann_stats_py",
    version="0.1.0",
    description="Analysis on Riemannian data.",
    long_description=readme(),
    long_description_content_type="text/x-rst",
    author="Oldemar Rodríguez Rojas, Jennifer Lobo Vásquez",
    author_email="oldemar.rodriguez@ucr.ac.cr, jennifer.lobo.vasquez@est.una.ac.cr",
    url="https://github.com/tu_usuario/riemann_stats_py",  # Actualizá este valor
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.9.2,<3.11",
        "pandas>=2.2.2,<2.3",
        "numpy>=1.26.4,<2.0",
        "scikit-learn>=1.5.1,<1.7",
        "umap-learn>=0.5.7,<0.6",
        "setuptools>=75.1.0,<80.0"
    ],
    python_requires=">=3.6",
    keywords=[
        "Riemannian Manifold",
        "Riemannian Principal Component Analysis (R-PCA)",
        "Riemannian Statistics",
        "Local Distance Notion",
        "Dimension Reduction",
        "Geometric Structures"
    ],
    classifiers=[
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
        "Operating System :: POSIX :: Linux",
    ],
    license="BSD-3-Clause",
    extras_require={
        "plot": [
            "matplotlib>=3.9.2,<3.11",
            "pandas>=2.2.2,<2.3"
        ]
    },
    test_suite="pytest",
    tests_require=["pytest"],
    zip_safe=False,
)
