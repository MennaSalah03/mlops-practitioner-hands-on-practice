import shutil
from pathlib import Path

import mlflow
import structlog

from prodml.config import config

logger = structlog.get_logger()


def export_best_model_from_mlflow() -> None:
    """Fetches the best run from MLflow and exports ONNX + DictVectorizer artifacts for production."""
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)

    # Search for the run with the lowest RMSE
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError("No runs found in MLflow experiment.")

    best_run_id = runs.iloc[0].run_id
    logger.info("best_run_found", run_id=best_run_id, rmse=runs.iloc[0]["metrics.rmse"])

    # Download ONNX model and DictVectorizer from MLflow artifacts
    onnx_path = mlflow.artifacts.download_artifacts(
        run_id=best_run_id, artifact_path="model/model.onnx"
    )
    dv_path = mlflow.artifacts.download_artifacts(
        run_id=best_run_id, artifact_path="dv.pkl"
    )

    # Copy to destination paths configured for Litestar API serving
    target_onnx = Path(config.onnx_model_path)
    target_dv = Path(config.pickle_model_path)

    target_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, target_onnx)
    shutil.copy(dv_path, target_dv)

    logger.info(
        "production_artifacts_exported", onnx=str(target_onnx), dv=str(target_dv)
    )


if __name__ == "__main__":
    export_best_model_from_mlflow()
