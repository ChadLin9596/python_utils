from setuptools import setup, find_packages

description = """
A few utility functions for Robotics, Computer Vision, Machine Learning
"""

setup(
    name="py_utils",
    version="0.1.0",
    description=description,
    author="Chad Lin",
    url="https://github.com/ChadLin9596/python_utils",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "dill",
        "lzf",
        "matplotlib",
        "numpy",
        "pandas",
        "Pillow",
        "requests",
        "scipy",
        "torch",
        "torchvision",
    ],
)
