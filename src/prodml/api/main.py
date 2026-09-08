"""Litestar API for the DurationPredictor model."""

import json
from pathlib import Path

import structlog
from anyio import to_thread
from litestar import Litestar, get, post
from litestar.di import NamedDependency, Provide
from litestar.exceptions import HTTPException
from litestar.response import Redirect

from prodml.api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from prodml.config import config
from prodml.predict import BaseModelPredictor, DurationPredictor

logger = structlog.get_logger().bind(component="api")


# ── Dependency factory (composition root) ────────────
def get_predictor() -> BaseModelPredictor:
    """Litestar calls this once at startup (use_cache=True below) and loads
    the model artifact so every request reuses the same in-memory predictor.

    DurationPredictor.load() reads the vectorizer from
    config.pickle_model_path and the regression graph from
    config.onnx_model_path — no paths passed here, so those two config
    values stay the single source of truth (see predict.py).
    """
    logger.info("initializing_dependencies")
    return DurationPredictor().load()


# ── Handlers ──────────────────────────────────────────
@post("/predict", description="upload a single trip data to get duration prediction")
async def predict(
    data: PredictRequest,
    predictor: NamedDependency[BaseModelPredictor],
) -> PredictResponse:
    """Predict trip duration for a single trip."""
    logger.info("prediction_request_received")

    record = data.to_record()
    result = predictor.predict_one(record)

    logger.info("prediction_request_successful", result=result)
    return PredictResponse(prediction_minutes=result)


@post("/predict/batch", description="upload batch trip data to get duration prediction")
async def predict_batch(
    data: PredictBatchRequest,
    predictor: NamedDependency[BaseModelPredictor],
) -> PredictBatchResponse:
    """Predict trip duration for a batch of trips."""
    logger.info("batch_prediction_request_received", count=len(data.trips))

    records = data.to_records()
    results = predictor.predict_batch(records)

    logger.info("batch_prediction_request_successful", count=len(results))
    return PredictBatchResponse(predictions_minutes=results)


@get("/")
async def root() -> Redirect:
    return Redirect(path="/health")


@get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


def _read_metadata_sync(path: Path) -> dict:
    """The actual blocking I/O for getting metadata from JSON file."""
    with open(path) as f_in:
        return json.load(f_in)


@get("/metadata")
async def metadata() -> MetadataResponse:
    path = Path(config.metadata_path)

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model metadata not found - model may not exist",
        )

    data = await to_thread.run_sync(_read_metadata_sync, path)

    return MetadataResponse(**data)


# ── App ───────────────────────────────────────────────
app = Litestar(
    route_handlers=[root, predict, predict_batch, health, metadata],
    dependencies={
        "predictor": Provide(get_predictor, use_cache=True, sync_to_thread=False),
    },
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.api_host, port=config.api_port)
