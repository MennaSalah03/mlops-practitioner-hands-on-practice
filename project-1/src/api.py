""" Litestar API for our model """

from litestar import Litestar, post, get
from litestar.di import Provide
from pydantic import BaseModel
import structlog
import logging

from src.logger import configure_logging
from src.model import DiabetesSklearnModel
from src.cache import RedisCacheManager
from src.service import InferenceService

# ── 1. Schema (The Web Boundary) ─────────────────────
class PredictRequest(BaseModel):
    """ The diabetes feature class """
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float

    def to_list(self) -> list[float]:
        """The bridge between the Web Domain and the Math Domain."""
        return [self.age, self.sex, self.bmi, self.bp, self.s1, self.s2]


configure_logging()
logger = structlog.get_logger().bind(component="api")

class PredictResponse(BaseModel):
    """ predict response """
    prediction: float
    status: str = "ok"

# ── 2. Dependency Factory (Composition in Action) ────
def get_inference_service() -> InferenceService:
    """
    Litestar runs this exactly once at startup. 
    It builds the pieces and wires them together.
    """
    # 1. Build the math engine
    logger.info("initializing_dependencies")
    predictor = DiabetesSklearnModel(model_path="models/diabetes_model.joblib")

    # 2. Build the cache memory
    # (Using "localhost" for local dev, will change to "redis" in Docker)
    cache = RedisCacheManager(host="localhost", port=6379)

    # 3. Compose them into the service
    return InferenceService(predictor=predictor, cache=cache)


# ── 3. Handlers (The API Logic) ──────────────────────
@post("/predict")
async def predict(
    data: PredictRequest,
    service: InferenceService,  # Litestar injects the composed service here
) -> PredictResponse:
    """ prediction posted to the API """
    logger.info("prediction_request_received")
    # 1. Strip the JSON labels to get the raw numbers
    ordered_features = data.to_list()

    # 2. Ask the service (which handles both Redis and the ML model)
    result = service.get_prediction(ordered_features)

    logger.info("prediction_request_successful", result=result)

    # 3. Return the formatted response
    return PredictResponse(prediction=result)


@get("/health")
async def health() -> dict:
    """ health endpoint """
    return {"status": "healthy"}


# ── 4. App Initialization ────────────────────────────
app = Litestar(
    route_handlers=[predict, health],
    dependencies={
        # use_cache=True guarantees the factory only runs once.
        "service": Provide(get_inference_service, use_cache=True, sync_to_thread=False)
    },
)

if __name__ == "__main__":
    import uvicorn
    # uvicorn app:app --reload (if running from terminal)
    uvicorn.run(app, host="127.0.0.1", port=8001)
