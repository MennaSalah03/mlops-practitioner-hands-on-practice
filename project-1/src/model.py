""" The model prediction class """

from abc import ABC, abstractmethod
import structlog
import joblib

logger = structlog.get_logger()

class BaseModelPredictor(ABC):
    """" model prediction abstaction class """
    @abstractmethod
    def predict(self, features: list[float]) -> float: ...

class DiabetesSklearnModel(BaseModelPredictor):
    """ Prediction class for the Diabetes model """
    def __init__(self, model_path: str) -> None:
        self.logger = logger.bind(component="model", model_path=model_path)

        self.logger.info("loading_model_started")
        self._model = joblib.load(model_path)
        self.logger.info("loading_model_complete")

    def predict(self, features: list[float]):
        self.logger.debug("prediction_execution_started", feature_count=len(features))

        result = float(self._model.predict([features])[0])

        self.logger.debug("prediction_execution_complete", prediction=result)
        return result
