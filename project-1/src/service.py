""" Inference """

import hashlib
import json
from src.model import BaseModelPredictor
from src.cache import BaseCacheManager

class InferenceService:
    """ Inference of MaseModelPredicitor class and the BaseCacheManager class """
    def __init__(self, predictor: BaseModelPredictor, cache: BaseCacheManager):
        # Composition: The service "has a" predictor and "has a" cache.
        self.predictor = predictor
        self.cache = cache

    def _generate_cache_key(self, features: list[float]) -> str:
        """Hashes the exact input features to create a unique lookup string."""
        feature_str = json.dumps(features)
        return hashlib.md5(feature_str.encode()).hexdigest()

    def get_prediction(self, features: list[float]) -> float:
        cache_key = self._generate_cache_key(features)
        
        # Ask the cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            # In a real app, use structlog here instead of print
            print(f"CACHE HIT: {cache_key}")
            return cached_result

        # Ask the model
        print(f"CACHE MISS: Running inference for {cache_key}")
        prediction = self.predictor.predict(features)

        # Save for next time
        self.cache.set(cache_key, prediction)

        return prediction
