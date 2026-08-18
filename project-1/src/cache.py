""" Cache """

from abc import ABC, abstractmethod
import redis

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
        self.client = redis.Redis(host=host, port=port, decode_responses=True)

    def get(self, key: str) -> float | None:
        result = self.client.get(key)
        return float(result) if result else None

    def set(self, key: str, value: float, ttl: int = 3600) -> None:
        # setex sets the value and the Time-To-Live (expiration) simultaneously
        self.client.setex(name=key, time=ttl, value=value)
