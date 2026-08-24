"""Load the model artifact and serve predictions."""

import pickle
from abc import ABC, abstractmethod

import structlog

from prodml.config import config
from prodml.decorators import timed
from prodml.features import prepare_inference_record

logger = structlog.get_logger()


class BaseModelPredictor(ABC):
    """Prediction Interface that API, BentoML runner and ONNX model all sit behind."""

    @abstractmethod
    def load(self, model_path: str | None = None) -> "BaseModelPredictor": ...

    @abstractmethod
    def predict_one(self, record: dict) -> float: ...

    @abstractmethod
    def predict_batch(self, records: list[dict]) -> list[float]: ...


class DurationPredictor(BaseModelPredictor):
    def __init__(self) -> None:
        self._dv = None
        self._model = None
        self.logger = logger.bind(component="DurationPredictor")

    def load(self, model_path: str | None = None) -> "DurationPredictor":
        path = model_path or config.model_path
        self.logger.info("loading_model_started", path=path)
        with open(path, "rb") as f_in:
            artifact = pickle.load(f_in)
        self._dv = artifact["dv"]
        self._model = artifact["model"]
        self.logger.info("loading_model_complete", path=path)
        return self

    def _ensure_loaded(self) -> None:
        if self._model is None or self._dv is None:
            raise RuntimeError("Model not loaded — call .load() first.")

    @timed
    def predict_one(self, record: dict) -> float:
        self._ensure_loaded()
        feature_dict = prepare_inference_record(record)
        X = self._dv.transform([feature_dict])
        result = float(self._model.predict(X)[0])
        self.logger.debug("predict_one", prediction=result)
        return result

    def predict_batch(self, records: list[dict]) -> list[float]:
        self._ensure_loaded()
        feature_dicts = [prepare_inference_record(r) for r in records]
        X = self._dv.transform(feature_dicts)
        results = self._model.predict(X).tolist()
        self.logger.debug("predict_batch", count=len(results))
        return results
