from setuptools import setup, Extension
import numpy as np
import sysconfig

# Remove -Wstrict-prototypes from default CFLAGS
cfg_vars = sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if isinstance(value, str) and "-Wstrict-prototypes" in value:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

ext_modules = [
    Extension(
        "numpy_reduceat_ext.argmin",
        sources=["src/numpy_reduceat_ext/argmin.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17",]
    ),
    Extension(
        "numpy_reduceat_ext.argmax",
        sources=["src/numpy_reduceat_ext/argmax.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17",]
    ),
]

setup(
    ext_modules=ext_modules,
)
