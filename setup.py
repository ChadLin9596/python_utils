from setuptools import setup, Extension
import numpy as np

ext_modules = [
    Extension(
        "numpy_reduceat_ext._argmin",
        sources=["src/numpy_reduceat_ext/argmin.c"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "numpy_reduceat_ext._argmax",
        sources=["src/numpy_reduceat_ext/argmax.c"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=ext_modules,
)
