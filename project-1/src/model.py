""" The model prediction class """

from abc import ABC, abstractmethod
import joblib


class BaseModelPredictor(ABC):
    """" model prediction abstaction class """
    @abstractmethod
    def predict(self, features: list[float]) -> float: ...

class DiabetesSklearnModel(BaseModelPredictor):
    """ Prediction class for the Diabetes model """
    def __init__(self, model_path: str) -> None:
        self._model = joblib.load(model_path)
    def predict(self, features: list[float]):
        return float(self._model.predict([features])[0])
