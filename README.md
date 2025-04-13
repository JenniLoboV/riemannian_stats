
---

# riemann_stats_py

**riemann_stats_py** is an open-source package that implements advanced methods for principal component analysis on Riemannian data using UMAP. It provides integrated tools for data preprocessing, statistical analysis, and visualization—making it an ideal choice for data science and machine learning projects that require extracting and representing key features in Riemannian spaces.

---

## 📦 Estructura del Paquete

The project structure is organized as follows:

```
riemann_stats_py/
│
├── riemann_stats_py/
│   ├── __init__.py                      # Makes package modules importable
│   ├── data_processing.py               # Classes for data loading and manipulation
│   ├── riemannian_umap_analysis.py      # Riemannian statistical methods using UMAP
│   ├── visualization.py                 # Functions and classes for result visualization
│   └── utilities.py                     # General utility functions
│
├── tests/                               # Unit tests for each module
│   ├── __init__.py
│   ├── test_riemannian_umap_analysis.py
│   ├── test_visualization.py
│   └── test_utilities.py
│
├── docs/                                # Project documentation
│   └── ...
│
├── examples/                            # Examples demonstrating package usage
│   ├── data/
│       └── Data10D_250.cvs
│       └── iris.cvs
│   ├── example1.py
│   └── example2.py
│
├── requirements.txt                     # Dependencies 
├── pyproject.toml                       # Package installation script
├── README.md                            # General information and usage of the package
└── LICENSE                              # BSD-3-Clause License

```

---

## 🚀 Installation

Ensure you have [Python ≥ 3.6](https://www.python.org/downloads/) installed, then run:

```bash
pip install riemann_stats_py
```

Alternatively, to install from the source code, clone the repository and execute:

```bash
git clone https://github.com/tu_usuario/riemann_stats_py.git
cd riemann_stats_py
pip install .
```

This project follows PEP 621 and uses pyproject.toml as the primary configuration file.

**Main Dependencies:**

- **matplotlib** (>=3.9.2,<3.11)
- **pandas** (>=2.2.2,<2.3)
- **numpy** (>=1.26.4,<2.0)
- **scikit-learn** (>=1.5.1,<1.7)
- **umap-learn** (>=0.5.7,<0.6)
- **setuptools** (>=75.1.0,<80.0)

These dependencies are defined in the [pyproject.toml](./pyproject.toml) and in [requirements.txt](./requirements.txt) .

---

## 🛠️ Features and Usage

**riemann_stats_py** offers several key functionalities:

- **Data Preprocessing:**  
  Easily import and transform datasets using functions in `data_processing.py`.

- **Riemannian Analysis with UMAP:**  
  Perform advanced statistical methods with `riemannian_umap_analysis.py` for extracting principal components in Riemannian spaces.

- **Visualization:**  
  Generate insightful 2D and 3D plots, along with other visualizations using `visualization.py`.

- **Additional Utilities:**  
  Use helper functions available in `utilities.py` for various tasks.

---

## 📚 Examples

The `examples/` directory contains two comprehensive examples demonstrating how to leverage **riemann_stats_py** for Riemannian data analysis and visualization.

### Example 1: Data10D_250 Dataset

This example demonstrates the analysis of a high-dimensional dataset (`Data10D_250.csv`). The workflow includes:

- **Data Loading and Preprocessing:**  
  The dataset is loaded with `DataProcessing.load_data()`, using a comma as the separator and a dot for decimals. If a `cluster` column exists, clustering information is separated from the main analysis data, while retaining a copy for visualization.

- **UMAP and Riemannian Analysis:**  
  An instance of `RiemannianUMAPAnalysis` is created with a neighbor count calculated as the dataset length divided by 5. The analysis includes:
  - UMAP graph similarities.
  - Computation of the rho matrix.
  - Calculation of Riemannian vector differences.
  - Generation of the UMAP distance matrix.
  - Derivation of Riemannian covariance and correlation matrices.
  - Extraction of principal components from the correlation matrix.
  - Calculation of explained inertia using the first two principal components.

- **Visualization:**  
  Depending on the presence of clustering data, the example produces:
  - A **2D scatter plot** with clusters.
  - A **Principal plane plot** showcasing principal components.
  - A **3D scatter plot** with clusters.
  - A **Correlation circle plot** to display correlations between original variables and principal components.

*For full details, see [example1.py](./examples/example1.py)*

---

### Example 2: Iris Dataset

Using the classic Iris dataset (`iris.csv`), this example illustrates the package's capabilities on a well-known, lower-dimensional dataset:

- **Data Loading and Preprocessing:**  
  The Iris dataset is imported using `DataProcessing.load_data()`, with a semicolon as the separator and a dot as the decimal. It checks for a `tipo` column to extract clustering information, which is then separated from the analysis data.

- **UMAP and Riemannian Analysis:**  
  An instance of `RiemannianUMAPAnalysis` is initialized with the dataset and a neighbor count determined as the data length divided by 3. The analysis process includes:
  - Calculation of UMAP graph similarities.
  - Derivation of the rho matrix.
  - Computation of Riemannian vector differences.
  - Generation of the UMAP distance matrix.
  - Calculation of Riemannian covariance and correlation matrices.
  - Extraction of principal components.
  - Determination of explained inertia (as a percentage) using the first two components.
  - Evaluation of correlations between the original variables and principal components.

- **Visualization:**  
  When clustering data is available, the example generates:
  - A **2D scatter plot** with clusters (using dimensions such as `s.largo` and `s.ancho`).
  - A **Principal plane plot** with clusters.
  - A **3D scatter plot** with clusters (adding a third dimension with `p.largo`).
  - A **Correlation circle plot** that is produced in all cases.

*For full details, see [example2.py](./examples/example2.py)*

---

## 🔍 Testing

The package includes a suite of unit tests located in the `tests/` directory. To run the tests, ensure [pytest](https://pytest.org/) is installed (it's included in the package configuration) and run:

```bash
pytest
```

This ensures that all functions and modules perform as expected throughout development and maintenance.

---

## 👥 Authors & Contributors

- **Oldemar Rodríguez Rojas** – Developed the mathematical functions and conducted the research.
- **Jennifer Lobo Vásquez** – Led the overall development and integration of the package.

---

## 📄 License

Distributed under the BSD-3-Clause License. See the [LICENSE](./LICENSE.txt) for more details.

---

## ❓ Support & Contributions

If you encounter any issues or have suggestions for improvements, please open an issue on the repository or submit a pull request. Your feedback is invaluable to enhancing the package.

---
## 📚 References

- **[Matplotlib Documentation](https://matplotlib.org/stable/contents.html)**  
  Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.  
  📦 PyPI: [matplotlib · PyPI](https://pypi.org/project/matplotlib/)

- **[Pandas Documentation](https://pandas.pydata.org/docs/)**  
  Pandas provides high-performance, easy-to-use data structures and data analysis tools for Python.  
  📦 PyPI: [pandas · PyPI](https://pypi.org/project/pandas/)

- **[NumPy Documentation](https://numpy.org/doc/)**  
  NumPy is the fundamental package for numerical computation in Python.  
  📦 PyPI: [numpy · PyPI](https://pypi.org/project/numpy/)

- **[Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)**  
  Scikit-learn is a machine learning library for Python, providing tools for classification, regression, clustering, and dimensionality reduction.  
  📦 PyPI: [scikit-learn · PyPI](https://pypi.org/project/scikit-learn/)

- **[UMAP-learn Documentation](https://umap-learn.readthedocs.io/)**  
  UMAP (Uniform Manifold Approximation and Projection) is a dimension reduction technique for visualization and general non-linear dimension reduction.  
  📦 PyPI: [umap-learn · PyPI](https://pypi.org/project/umap-learn/)

- **[Setuptools Documentation](https://setuptools.pypa.io/en/latest/)**  
  Setuptools is a package development and distribution tool used to package Python projects and manage dependencies.  
  📦 PyPI: [setuptools · PyPI](https://pypi.org/project/setuptools/)

---

```