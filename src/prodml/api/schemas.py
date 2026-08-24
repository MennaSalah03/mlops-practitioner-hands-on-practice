"""Pydantic schemas — the web boundary. Translates HTTP JSON <-> domain dicts."""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Raw trip fields, as a caller (booking app, dispatcher, etc.) would know them."""

    PULocationID: int
    DOLocationID: int
    trip_distance: float

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
    status: str = "ok"


class PredictBatchResponse(BaseModel):
    predictions_minutes: list[float]
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str = "healthy"
