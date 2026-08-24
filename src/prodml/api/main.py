"""Litestar API for the DurationPredictor model."""

import structlog
from litestar import Litestar, get, post
from litestar.di import Provide

from prodml.api.schemas import (
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from prodml.config import config
from prodml.predict import DurationPredictor

logger = structlog.get_logger().bind(component="api")


# ── Dependency factory (composition root) ────────────
def get_predictor() -> DurationPredictor:
    """Litestar calls this once at startup (use_cache=True below) and loads
    the model artifact so every request reuses the same in-memory predictor."""
    logger.info("initializing_dependencies")
    return DurationPredictor().load(config.model_path)


# ── Handlers ──────────────────────────────────────────
@post("/predict")
async def predict(
    data: PredictRequest,
    predictor: DurationPredictor,
) -> PredictResponse:
    """Predict trip duration for a single trip."""
    logger.info("prediction_request_received")

    record = data.to_record()
    result = predictor.predict_one(record)

    logger.info("prediction_request_successful", result=result)
    return PredictResponse(prediction_minutes=result)


@post("/predict/batch")
async def predict_batch(
    data: PredictBatchRequest,
    predictor: DurationPredictor,
) -> PredictBatchResponse:
    """Predict trip duration for a batch of trips."""
    logger.info("batch_prediction_request_received", count=len(data.trips))

    records = data.to_records()
    results = predictor.predict_batch(records)

    logger.info("batch_prediction_request_successful", count=len(results))
    return PredictBatchResponse(predictions_minutes=results)


@get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


# ── App ───────────────────────────────────────────────
app = Litestar(
    route_handlers=[predict, predict_batch, health],
    dependencies={
        "predictor": Provide(get_predictor, use_cache=True, sync_to_thread=False),
    },
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.api_host, port=config.api_port)
