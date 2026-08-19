""" Cache """

from abc import ABC, abstractmethod
import structlog
import redis

logger = structlog.get_logger()

class BaseCacheManager(ABC):
    """The strict contract for caching."""
    @abstractmethod
    def get(self, key: str) -> float | None:
        """ Abstraction class for cache getter method """

    @abstractmethod
    def set(self, key: str, value: float, ttl: int = 3600) -> None:
        """ Abstraction class for cache setter method """

class RedisCacheManager(BaseCacheManager):
    """The concrete implementation for Redis."""
    def __init__(self, host: str, port: int = 6379):
        # Decode responses ensures we get strings back, not bytes
        self.logger = logger.bind(component="cache", redis_host=host, redis_port=port)
        self.logger.info("initializing_redis_client")
        self.client = redis.Redis(host=host, port=port, decode_responses=True)

    def get(self, key: str) -> float | None:
        self.logger.debug("redis_get_attempt", key=key)
        result = self.client.get(key)
        return float(result) if result else None

    def set(self, key: str, value: float, ttl: int = 3600) -> None:
        self.logger.debug("redis_set_attempt", key=key, ttl=ttl)
        self.client.setex(name=key, time=ttl, value=value)
