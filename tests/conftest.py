from datetime import datetime, timedelta

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_feature_dicts() -> list[dict]:
    """Pre-vectorization feature dicts — the output shape of to_feature_dicts()"""
    return [
        {"PU_DO": "100_10", "trip_distance": 5.0},
        {"PU_DO": "101_132", "trip_distance": 12.0},
        {"PU_DO": "100_18", "trip_distance": 8.0},
        {"PU_DO": "102_130", "trip_distance": 3.5},
    ]


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Raw DataFrame containing normal records, outliers, and missing values."""
    base_time = datetime.now()  # noqa: DTZ005

    data = [
        # Normal trip: 15 mins, standard IDs
        {
            "lpep_pickup_datetime": base_time,
            "lpep_dropoff_datetime": base_time + timedelta(minutes=15),
            "PULocationID": 100,
            "DOLocationID": 10,
            "trip_distance": 5.0,
        },
        # Outlier (too short): 0 mins
        {
            "lpep_pickup_datetime": base_time,
            "lpep_dropoff_datetime": base_time,
            "PULocationID": 101,
            "DOLocationID": 132,
            "trip_distance": 12.0,
        },
        # Outlier (too long): 100 mins
        {
            "lpep_pickup_datetime": base_time,
            "lpep_dropoff_datetime": base_time + timedelta(minutes=100),
            "PULocationID": 102,
            "DOLocationID": 130,
            "trip_distance": 3.5,
        },
        # Edge Case: Missing location IDs (NaN), zero distance
        {
            "lpep_pickup_datetime": base_time,
            "lpep_dropoff_datetime": base_time + timedelta(minutes=10),
            "PULocationID": "nan",
            "DOLocationID": "nan",
            "trip_distance": 0.0,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def mock_config(monkeypatch):
    """Mocks the config object used in features.py for deterministic tests."""

    class MockConfig:
        target = "duration"
        min_duration = 1.0
        max_duration = 60.0
        categorical = ["PU_DO"]  # noqa: RUF012
        numerical = ["trip_distance"]  # noqa: RUF012

    monkeypatch.setattr("prodml.features.config", MockConfig)
    return MockConfig


@pytest.fixture
def fitted_artifact(sample_raw_dataframe):
    """Trains a model on the dummy DataFrame to generate a real artifact in memory."""
    from prodml.features import feature_engineering
    from prodml.train import train_model

    # We pass the sample dataframe as both train and test just to get a fitted model
    df_train, df_test = feature_engineering(
        sample_raw_dataframe, sample_raw_dataframe.copy()
    )
    artifact = train_model(df_train, df_test)
    return artifact


@pytest.fixture
def exported_models(fitted_artifact, monkeypatch, tmp_path):
    """Exports both ONNX and Pickle versions to a safe, disposable directory."""
    from prodml.config import config
    from prodml.export import persist_model_onnx, persist_model_pickle

    # 1. Define paths inside Pytest's temporary folder (/tmp/pytest-of-mennasalah/...)
    pickle_model_path = tmp_path / "baseline.pkl"
    onnx_model_path = tmp_path / "baseline.onnx"
    metadata_path = tmp_path / ".metadata.json"

    # 2. Intercept the Pydantic config object and overwrite the paths
    monkeypatch.setattr(config, "pickle_model_path", str(pickle_model_path))
    monkeypatch.setattr(config, "onnx_model_path", str(onnx_model_path))
    monkeypatch.setattr(config, "metadata_path", str(metadata_path))

    # 3. Run the actual export functions (they will now write to tmp_path)
    persist_model_onnx(fitted_artifact, str(onnx_model_path))
    persist_model_pickle(fitted_artifact, str(pickle_model_path))

    return {"onnx": str(onnx_model_path), "pkl": str(pickle_model_path)}
