"""Load the model artifact and serve predictions."""

import pickle
from abc import ABC, abstractmethod

import mlflow
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
    self,
    model_uri: str | None = None,
    dv_artifact_path: str = "dv.pkl",
) -> "MLflowDurationPredictor":
    model_uri = model_uri or config.mlflow_model_uri

    self.logger.info(
        "loading_onnx_model_started",
        model_uri=model_uri,
    )

    try:
        onnx_model = mlflow.onnx.load_model(model_uri)

        self.logger.info(
            "onnx_model_downloaded",
            model_uri=model_uri,
        )

        self._session = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )

        self._input_name = self._session.get_inputs()[0].name

        self.logger.info(
            "onnx_runtime_initialized",
            input_name=self._input_name,
        )

    except Exception:
        self.logger.exception(
            "onnx_model_loading_failed",
            model_uri=model_uri,
        )
        raise

    try:
        run_id = mlflow.models.get_model_info(model_uri).run_id

        self.logger.info(
            "loading_vectorizer_started",
            run_id=run_id,
            artifact_path=dv_artifact_path,
        )

        dv_local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=dv_artifact_path,
        )

        with open(dv_local_path, "rb") as f_in:
            self._dv = pickle.load(f_in)

        self.logger.info(
            "vectorizer_loaded",
            run_id=run_id,
            path=dv_local_path,
        )

    except Exception:
        self.logger.exception(
            "vectorizer_loading_failed",
            model_uri=model_uri,
        )
        raise

    self.logger.info(
        "loading_model_complete",
        model_uri=model_uri,
        run_id=run_id,
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


class MLflowDurationPredictor(DurationPredictor):
    """Same predictor as DurationPredictor, but sources both artifacts from
    the MLflow Model Registry instead of local paths.

    The ONNX model is registered via mlflow.onnx.log_model, and the
    DictVectorizer is a separate artifact (dv.pkl) logged in the same run —
    NOT bundled into the model. So the pipeline is still explicitly:
    raw dict -> dv.transform() -> dense float32 -> onnxruntime.
    Only `load()` differs from DurationPredictor; predict_one/predict_batch
    are inherited unchanged.
    """

    def load(
        self,
        model_uri: str | None = None,
        dv_artifact_path: str = "dv.pkl",
    ) -> "MLflowDurationPredictor":
        # e.g. "models:/ride-duration-predictor/Production" or ".../3"
        model_uri = model_uri or config.mlflow_model_uri

        self.logger.info("loading_onnx_model_started", model_uri=model_uri)
        onnx_model = mlflow.onnx.load_model(model_uri)
        self._session = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        run_id = mlflow.models.get_model_info(model_uri).run_id
        self.logger.info(
            "loading_vectorizer_started", run_id=run_id, artifact_path=dv_artifact_path
        )
        dv_local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=dv_artifact_path
        )
        with open(dv_local_path, "rb") as f_in:
            self._dv = pickle.load(f_in)

        self.logger.info("loading_model_complete", model_uri=model_uri, run_id=run_id)
        return self
