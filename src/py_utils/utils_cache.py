import json
import pickle
from functools import wraps
from pathlib import Path

import numpy as np


####################
# Cache Primitives #
####################


def save_cache(data, cache_path, serializer="pickle"):
    """Save data to a cache file. Creates parent directories as needed."""

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if serializer == "pickle":
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif serializer == "numpy":
        np.save(cache_path, data)

    elif serializer == "json":
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=4)

    else:
        raise ValueError(f"Unknown serializer: {serializer}")


def load_cache(cache_path, serializer="pickle"):
    """Load data from a cache file."""

    cache_path = Path(cache_path)

    if serializer == "pickle":
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    elif serializer == "numpy":
        result = np.load(cache_path)
        return result

    elif serializer == "json":
        with open(cache_path, "r") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unknown serializer: {serializer}")


def clear_cache(cache_path):
    """Remove a cache file if it exists."""

    cache_path = Path(cache_path)
    if cache_path.exists():
        cache_path.unlink()
        print(f"Cleared cache: {cache_path}")
    else:
        print(f"Cache not found: {cache_path}")


#####################
# cache_to_file      #
#####################


def cache_to_file(
    directory,
    filename,
    serializer="pickle",
):
    """Decorator that caches a function's return value to a file.

    Example — on a method::

        @cache_to_file(
            lambda self, idx: self.cache_root,
            lambda self, idx: f"result.{idx}.npy",
            serializer="numpy",
        )
        def get_result(self, idx):
            ...

    Example — on a plain function::

        @cache_to_file("/tmp/cache", "result.pkl")
        def expensive():
            ...
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            dir_path = directory
            if callable(directory):
                dir_path = directory(*args, **kwargs)

            file_name = filename
            if callable(filename):
                file_name = filename(*args, **kwargs)

            cache_path = Path(dir_path) / file_name

            if cache_path.exists():
                return load_cache(cache_path, serializer)

            result = func(*args, **kwargs)
            save_cache(result, cache_path, serializer)
            return result

        return wrapper

    return decorator
