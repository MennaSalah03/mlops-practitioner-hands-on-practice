"""Pydantic schemas — the web boundary. Translates HTTP JSON <-> domain dicts."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Raw trip fields, as a caller (booking app, dispatcher, etc.) would know them."""

    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    trip_distance: float = Field(gt=0, lt=60)

    model_config = {
        "json_schema_extra": {
            "example": {"PULocationID": 43, "DOLocationID": 236, "trip_distance": 3.5}
        }
    }

    def to_record(self) -> dict:
        """Bridge between the web schema and the feature-engineering input shape."""
        return self.model_dump()


class PredictBatchRequest(BaseModel):
    """A batch of trips to score in one call."""

    trips: list[PredictRequest]

    def to_records(self) -> list[dict]:
        return [trip.to_record() for trip in self.trips]


class PredictResponse(BaseModel):
    prediction_minutes: float
    model_version: str = "unknown"
    correlation_id: UUID = Field(default_factory=uuid4)
    latency_ms: float = 0.0
    status: str = "ok"


class PredictBatchResponse(BaseModel):
    predictions_minutes: list[float]
    model_version: str = "unknown"
    correlation_id: UUID = Field(default_factory=uuid4)
    latency_ms: float = 0.0
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True


class MetadataResponse(BaseModel):
    model_version: str
    training_date: datetime
    feature_names: list[str]
    framework: str
    metrics: dict
    artifact_hash: str
