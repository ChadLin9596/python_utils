Installation
============

Install from GitHub
-------------------

.. code-block:: bash

    $ pip install git+https://github.com/ChadLin9596/python_utils.git

Custom Installation
-------------------

``python_utils`` is modular. Each submodule has its own optional
dependencies. You only need to install the dependencies for the parts you
use.

Check the full module-level dependency list here:

:doc:`Dependency Overview <dependency>`


Examples
--------

** Install only segmentation utilities:**

.. code-block:: bash

    pip install numpy_reduceat_ext numpy

then use

.. code-block:: python

    import py_utils.utils_segmentation


Notes
-----

- Importing a submodule does **not** pull dependencies from other modules.
- If you don’t import a heavy module (e.g., ``utils_torch``), its dependencies are never loaded.
- Each module’s required dependencies are documented at the top of its API page and in the :doc:`dependency` page.
