"Cache features from images"

from pathlib import Path
import torch
from .cache_utils import build_cache_path


class FeatureCacher:
    "Cache features from images"

    def __init__(self, cache_folder):
        self.cache_path = cache_folder
        self.create_cache_folder()

    def create_cache_folder(self):
        "Creates the full path for the cache folder"
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)

    def cache_feature(self, image_path, features):
        "Caches the features for a given image, considering its name"
        cache_file = build_cache_path(self.cache_path, image_path)
        torch.save(features, cache_file)


class CacheManager:
    """Manages a cache of features. If a feature is not cached
    (file not present in cache folder), it caches the feature
    and provides it. Else, it just loads from cache."""

    def __init__(self, cache_folder, feature_extractor):
        "Manages a cache of features"
        self.cache_folder = cache_folder
        self.feature_extractor = feature_extractor
        self.cached = set()
        self.feature_cacher = FeatureCacher(self.cache_folder)

    def get_features(self, image, image_path):
        """Get the image features from a cache, computes
        it if not present"""

        cache_path = build_cache_path(self.cache_folder, image_path)
        if self.is_cached(image_path):
            return torch.load(cache_path)
        # Extract the features
        features = self.feature_extractor.extract_features(image)
        self.feature_cacher.cache_feature(image_path, features)
        self.cached.add(image_path)
        return features

    def is_cached(self, image_path):
        "Checks if a image is already cached"
        if image_path in self.cached:
            return True
        # Checks filesystem only if not in buffer
        img_cached = build_cache_path(self.cache_folder, image_path).is_file()
        if img_cached:
            self.cached.add(image_path)
        return img_cached
