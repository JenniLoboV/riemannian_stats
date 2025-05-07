How to Install It
==================

To install the **RiemannStats** package, you can use **pip** from the Python Package Index (PyPI) or install it directly from the source.

Install via PyPI
----------------

To install the latest version of **RiemannStats** directly from PyPI, simply run the following command:

.. code-block:: bash

   pip install riemannstats

This will automatically install the package along with its required dependencies.

Install from Source
-------------------

If you prefer to install **RiemannStats** from the source code, follow these steps:

1. Clone the repository from GitHub:

.. code-block:: bash

   git clone https://github.com/your_username/riemann_stats_py.git

2. Navigate to the cloned directory:

.. code-block:: bash

   cd riemann_stats_py

3. Install the package and its dependencies:

.. code-block:: bash

   pip install .

Requirements
-------------
Make sure you have the necessary dependencies listed in the `requirements.txt` file. The core dependencies include:

- **matplotlib** (>=3.9.2,<3.11)
- **pandas** (>=2.2.2,<2.3)
- **numpy** (>=1.26.4,<2.0)
- **scikit-learn** (>=1.5.1,<1.7)
- **umap-learn** (>=0.5.7,<0.6)
- **setuptools** (>=75.1.0,<80.0)

These dependencies are automatically installed when you use `pip install riemannstats`. However, if you prefer to manage the dependencies manually, you can install them from the `requirements.txt` file:

.. code-block:: bash

   pip install -r requirements.txt

Python Version
---------------

**RiemannStats** requires Python version **3.8 or higher**. Ensure you have the correct version of Python installed. You can check your Python version by running:

.. code-block:: bash

   python --version

For more detailed installation instructions or to contribute to the project, visit the [GitHub repository](https://github.com/your_username/riemann_stats_py).
