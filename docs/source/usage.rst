Usage
=====

.. _installation:

Installation
------------

To use the `python_utils` package, you first need to install it. You can do this using pip:


.. code-block:: console

    $ git clone https://github.com/ChadLin9596/python_utils.git

    either
    $ pip install ./python_utils
    or
    $ pip install -e ./python_utils


Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. py:function:: lumache.get_random_ingredients(kind=None)

   Return a list of random ingredients as strings.

   :param kind: Optional "kind" of ingredients.
   :type kind: list[str] or None
   :return: The ingredients list.
   :rtype: list[str]


.. autofunction:: py_utils.utils_segmentation.segmented_sum
