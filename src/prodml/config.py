"""Centralized configuration — paths, hyperparameters, service settings"""

import tomllib
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_project_version() -> str:
    """Reads the version dynamically from pyproject.toml."""
    toml_path = Path("pyproject.toml")

    if not toml_path.exists():
        return "0.0.0-unknown"
    # tomllib requires files to be opened in binary mode ("rb")
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    return data.get("project", {}).get("version", "0.0.0-unknown")


class Settings(BaseSettings):
    """pydantic-settings configs"""

    # File paths with sane defaults
    data_path: str = "data/green_tripdata_2026-04.parquet"
    pickle_model_path: str = "models/baseline.pkl"
    onnx_model_path: str = "models/baseline.onnx"
    metadata_path: str = "models/.metadata.json"
    report_path: str = "reports/module-1.md"

    # Features & Hyperparameters
    raw_input_fields: list[str] = ["PULocationID", "DOLocationID", "trip_distance"]
    categorical: list[str] = ["PU_DO"]
    numerical: list[str] = ["trip_distance"]
    target: str = "duration"

    test_size: float = 0.2
    random_state: int = 42

    min_duration: int = 1
    max_duration: int = 60

    # read from a .env file if it exists
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    version: str = Field(default_factory=get_project_version)

    mlflow_tracking_uri: str
    postgres_user: str
    postgres_password: str
    postgres_db: str
    minio_root_user: str
    minio_root_password: str


# the single truth imported by the rest of the app.
config = Settings()
