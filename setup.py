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
