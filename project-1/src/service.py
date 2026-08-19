""" Inference """

import hashlib
import json
import structlog
import logging
from src.cache import BaseCacheManager
from src.model import BaseModelPredictor

logger = structlog.get_logger()

class InferenceService:
    """ Inference of MaseModelPredicitor class and the BaseCacheManager class """
    def __init__(self, predictor: BaseModelPredictor, cache: BaseCacheManager):
        # Composition: The service "has a" predictor and "has a" cache.
        self.predictor = predictor
        self.cache = cache
        self.logger = logger.bind(component="inference_service")

    def _generate_cache_key(self, features: list[float]) -> str:
        """Hashes the exact input features to create a unique lookup string."""
        feature_str = json.dumps(features)
        return hashlib.md5(feature_str.encode()).hexdigest()

    def get_prediction(self, features: list[float]) -> float:
        """ get prediction """
        cache_key = self._generate_cache_key(features)

        # Ask the cache if it has the prediction already stored
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.info("cache_hit", key=cache_key)
            return cached_result

        # Ask the model
        self.logger.info("cache_miss", key=cache_key, action="running_inference")
        prediction = self.predictor.predict(features)

        # Save for next time
        self.logger.debug("saving_to_cache", key=cache_key)
        self.cache.set(cache_key, prediction)

        return prediction
