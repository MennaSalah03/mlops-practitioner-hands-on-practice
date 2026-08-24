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
from prodml.cache import RedisCacheManager
from prodml.config import config
from prodml.predict import DurationPredictor
from prodml.service import InferenceService

logger = structlog.get_logger().bind(component="api")


# ── Dependency factory (composition root) ────────────
def get_inference_service() -> InferenceService:
    """
    Litestar calls this once at startup (use_cache=True below) and wires
    the predictor + cache into one InferenceService for handlers to use.
    """
    logger.info("initializing_dependencies")

    predictor = DurationPredictor().load(config.model_path)
    cache = RedisCacheManager(host=config.redis_host, port=config.redis_port)

    return InferenceService(predictor=predictor, cache=cache)


# ── Handlers ──────────────────────────────────────────
@post("/predict")
async def predict(
    data: PredictRequest,
    service: InferenceService,
) -> PredictResponse:
    """Predict trip duration for a single trip."""
    logger.info("prediction_request_received")

    record = data.to_record()
    result = service.get_prediction(record)

    logger.info("prediction_request_successful", result=result)
    return PredictResponse(prediction_minutes=result)


@post("/predict/batch")
async def predict_batch(
    data: PredictBatchRequest,
    service: InferenceService,
) -> PredictBatchResponse:
    """Predict trip duration for a batch of trips (bypasses cache, direct to model)."""
    logger.info("batch_prediction_request_received", count=len(data.trips))

    records = data.to_records()
    results = service.predictor.predict_batch(records)

    logger.info("batch_prediction_request_successful", count=len(results))
    return PredictBatchResponse(predictions_minutes=results)


@get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


# ── App ───────────────────────────────────────────────
app = Litestar(
    route_handlers=[predict, predict_batch, health],
    dependencies={
        "service": Provide(get_inference_service, use_cache=True, sync_to_thread=False),
    },
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.api_host, port=config.api_port)
