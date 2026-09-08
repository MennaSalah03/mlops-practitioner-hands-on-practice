"""Integration tests for the Litestar API (prodml.api.main).

ASSUMPTION FLAGGED: I don't have your real prodml/api/schemas.py, so these
tests assume the following shapes, inferred from how main.py uses them:
    PredictRequest       -> PULocationID: int, DOLocationID: int, trip_distance: float
    PredictBatchRequest  -> trips: list[PredictRequest]
    PredictResponse      -> prediction_minutes: float
    PredictBatchResponse -> predictions_minutes: list[float]
    HealthResponse       -> status: str
    MetadataResponse     -> mirrors the metadata dict's keys exactly
If your real schemas.py differs, the request payloads below will fail
validation — send me schemas.py and I'll align these exactly.

Uses the real app (routes + DI) via api_client (conftest.py), which is built
on exported_models — real onnx + pickle files on disk, real
DurationPredictor, real onnxruntime inference. Nothing here is mocked.
"""

import json

import pytest


def test_root_redirects_to_health(api_client):
    response = api_client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert response.headers["location"] == "/health"


def test_health_returns_200(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# /metadata
# --------------------------------------------------------------------------- #
def test_metadata_returns_expected_fields(api_client, exported_models):
    from prodml.config import config

    with open(config.metadata_path) as f_in:
        expected = json.load(f_in)

    response = api_client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == expected["model_version"]
    assert body["artifact_hash"] == expected["artifact_hash"]


def test_metadata_returns_503_when_file_missing(exported_models, monkeypatch, tmp_path):
    """Points metadata_path at a file that doesn't exist — the endpoint
    should fail with 503 (service not ready), not a raw 500 from json.load."""
    from litestar.di import Provide
    from litestar.testing import create_test_client

    from prodml.api.main import (
        get_predictor,
        health,
        metadata,
        predict,
        predict_batch,
        root,
    )
    from prodml.config import config

    monkeypatch.setattr(config, "metadata_path", str(tmp_path / "does_not_exist.json"))

    with create_test_client(
        route_handlers=[root, predict, predict_batch, health, metadata],
        dependencies={
            "predictor": Provide(get_predictor, use_cache=True, sync_to_thread=False)
        },
    ) as client:
        response = client.get("/metadata")
        assert response.status_code == 503


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def test_predict_returns_200_and_float(api_client, sample_inference_record):
    response = api_client.post("/predict", json=sample_inference_record)
    assert response.status_code == 201  # Litestar's default @post success code
    body = response.json()
    assert isinstance(body["prediction_minutes"], float)


def test_predict_matches_sklearn_baseline(
    api_client, fitted_artifact, sample_inference_record
):
    """Confirms the API's onnxruntime-backed prediction agrees with the
    sklearn model on the exact same request — catches silent divergence
    from ONNX conversion at the API layer, not just in isolation."""
    response = api_client.post("/predict", json=sample_inference_record)
    api_result = response.json()["prediction_minutes"]

    dv = fitted_artifact["dv"]
    model = fitted_artifact["model"]
    feature_dict = {
        "PU_DO": f"{sample_inference_record['PULocationID']}_{sample_inference_record['DOLocationID']}",
        "trip_distance": sample_inference_record["trip_distance"],
    }
    expected = float(model.predict(dv.transform([feature_dict]))[0])
    assert api_result == pytest.approx(expected, rel=1e-3)


def test_predict_missing_field_returns_400(api_client):
    """A malformed request (missing DOLocationID) should fail request
    validation before ever reaching the predictor. Litestar returns 400 for
    signature/validation failures (not 422, which is more of a FastAPI
    convention)."""
    incomplete_payload = {"PULocationID": 100, "trip_distance": 5.0}
    response = api_client.post("/predict", json=incomplete_payload)
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# /predict/batch
# --------------------------------------------------------------------------- #
def test_predict_batch_returns_200_and_matching_length(
    api_client, sample_inference_record
):
    payload = {
        "trips": [
            sample_inference_record,
            {"PULocationID": 101, "DOLocationID": 132, "trip_distance": 12.0},
        ]
    }
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert len(body["predictions_minutes"]) == 2
    assert all(isinstance(p, float) for p in body["predictions_minutes"])


def test_predict_batch_empty_trips_returns_empty_list(api_client):
    """Regression guard for the predict_batch([]) bug (DictVectorizer used
    to raise on an empty transform list) — should return [] cleanly."""
    response = api_client.post("/predict/batch", json={"trips": []})
    assert response.status_code == 201
    assert response.json()["predictions_minutes"] == []
