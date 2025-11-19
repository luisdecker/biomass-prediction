"Utilities for caching"

from pathlib import Path


def build_cache_path(cache_folder, image_path):
    "Build the cache path for a given image name"
    stem = Path(image_path).stem
    cache_filename = f"{stem}.pt"
    return Path(cache_folder, cache_filename)
