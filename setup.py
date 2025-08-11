from setuptools import setup, find_packages, Extension
import numpy as np

description = """
A few utility functions for Robotics, Computer Vision, Machine Learning
"""

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
    name="py_utils",
    version="0.1.0",
    description=description,
    author="Chad Lin",
    url="https://github.com/ChadLin9596/python_utils",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    install_requires=[
        "dill",
        "lzf",
        "matplotlib",
        "numpy<2",
        "opencv-python<4.12.0.88",
        "pandas",
        "Pillow",
        "requests",
        "scipy",
        "scikit-image",
        "torch",
        "torchvision",
    ],
)
