"""Load the model artifact and serve predictions."""

import pickle
from abc import ABC, abstractmethod

import numpy as np
import onnxruntime as ort
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
    """Runs the regression step through onnxruntime instead of sklearn.

    Still loads the DictVectorizer from the pickle artifact — export.py's
    persist_model_onnx only converts the model, not the vectorizer (a
    DictionaryType ONNX input only accepts raw Python dicts, not a float
    tensor — see that docstring), so `dv.transform(...)` still runs in
    Python here. Only the actual "predict the number" step is ONNX.
    """

    def __init__(self) -> None:
        self._dv = None
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self.logger = logger.bind(component="DurationPredictor")

    def load(
        self, model_path: str | None = None, dv_path: str | None = None
    ) -> "DurationPredictor":
        dv_source = dv_path or config.pickle_model_path
        self.logger.info("loading_vectorizer_started", path=dv_source)
        with open(dv_source, "rb") as f_in:
            artifact = pickle.load(f_in)
        self._dv = artifact["dv"]

        onnx_path = model_path or config.onnx_model_path
        self.logger.info("loading_onnx_model_started", path=onnx_path)
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        self.logger.info(
            "loading_model_complete", dv_path=dv_source, onnx_path=onnx_path
        )
        return self

    def _ensure_loaded(self) -> None:
        if self._dv is None or self._session is None:
            raise RuntimeError("Model not loaded — call .load() first.")

    def _to_dense_float32(self, X) -> np.ndarray:
        """dv.transform() returns a scipy sparse matrix; onnxruntime needs a
        dense float32 array."""
        return np.asarray(X.todense(), dtype=np.float32)

    @timed
    def predict_one(self, record: dict) -> float:
        self._ensure_loaded()
        feature_dict = prepare_inference_record(record)
        X = self._to_dense_float32(self._dv.transform([feature_dict]))
        result = float(self._session.run(None, {self._input_name: X})[0].ravel()[0])
        self.logger.debug("predict_one", prediction=result)
        return result

    def predict_batch(self, records: list[dict]) -> list[float]:
        self._ensure_loaded()
        if not records:
            return []
        feature_dicts = [prepare_inference_record(r) for r in records]
        X = self._to_dense_float32(self._dv.transform(feature_dicts))
        results = self._session.run(None, {self._input_name: X})[0].ravel().tolist()
        self.logger.debug("predict_batch", count=len(results))
        return results
