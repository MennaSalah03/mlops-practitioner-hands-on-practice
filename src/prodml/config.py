"""Centralized configuration — paths, hyperparameters, service settings"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """pydantic-settings configs"""

    # File paths with sane defaults
    data_path: str = "data/green_tripdata_2026-04.parquet"
    model_path: str = "models/baseline.pkl"
    report_path: str = "reports/module-1.md"

    # Features & Hyperparameters
    categorical: list[str] = ["PU_DO"]
    numerical: list[str] = ["trip_distance"]
    target: str = "duration"

    test_size: float = 0.2
    random_state: int = 42

    # read from a .env file if it exists
    min_duration: int = 1
    max_duration: int = 60
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# the single truth imported by the rest of the app.
config = Settings()
