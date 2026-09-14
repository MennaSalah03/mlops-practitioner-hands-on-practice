"""Train and log model runs to MLflow.

Three model families trained on identical splits: Linear Regression
(baseline), XGBoost, and a small PyTorch MLP. Each gets its own top-level
mlflow run. run_xgboost_sweep() runs an Optuna hyperparameter search over
XGBoost, with each trial logged as a nested run under one parent "sweep" run.
"""

import getpass
import io
import pickle
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.onnx
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import onnx
import optuna
import pandas as pd
import structlog
import torch
import torch.nn as nn  # noqa: PLR0402
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import (
    FloatTensorType as OnnxMLToolsFloatTensorType,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prodml.config import config
from prodml.features import to_feature_dicts

logger = structlog.get_logger()


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def setup_mlflow() -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)


# --------------------------------------------------------------------------- #
# Shared helpers — data, tags, plots, artifacts.
# --------------------------------------------------------------------------- #
def _to_onnx(model, model_type: str, n_features: int) -> onnx.ModelProto:
    """Converts a trained sklearn, xgboost, or pytorch model into an ONNX ModelProto.

    Three model families, three completely different conversion paths —
    this is the part of the file most worth slowing down on:
      - sklearn (LinearRegression): skl2onnx.convert_sklearn understands
        real sklearn estimators natively.
      - xgboost: XGBRegressor only *looks* sklearn-shaped (.fit/.predict);
        its internals are a different library, so skl2onnx has no shape
        calculator registered for it (that's the exact error you hit).
        onnxmltools ships the XGBoost-specific converter instead.
      - pytorch: neither of the above applies — torch has its own built-in
        ONNX exporter (torch.onnx.export), which traces the model's forward
        pass on a dummy input to build the graph.
    """
    if model_type == "sklearn":
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        return convert_sklearn(model, initial_types=initial_type, target_opset=12)

    elif model_type == "xgboost":
        # NOT skl2onnx — XGBRegressor isn't a real sklearn estimator
        # internally, so skl2onnx has no converter registered for it
        # ("Unable to find a shape calculator for type 'XGBRegressor'").
        # onnxmltools ships the XGBoost-specific converter.
        initial_type = [("float_input", OnnxMLToolsFloatTensorType([None, n_features]))]
        return convert_xgboost(model, initial_types=initial_type, target_opset=12)

    elif model_type == "pytorch":
        model.eval()
        dummy_input = torch.randn(1, n_features, dtype=torch.float32)
        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            dummy_input,
            buffer,
            input_names=["float_input"],
            output_names=["output"],
            dynamic_axes={
                "float_input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=12,
        )
        buffer.seek(0)
        return onnx.load(buffer)

    else:
        raise ValueError(f"Unsupported model type for ONNX conversion: {model_type}")


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _data_version_hash(df_train: pd.DataFrame, df_test: pd.DataFrame) -> str:
    train_hash = pd.util.hash_pandas_object(df_train, index=True).sum()
    test_hash = pd.util.hash_pandas_object(df_test, index=True).sum()
    return f"{train_hash & 0xFFFFFFFF:08x}{test_hash & 0xFFFFFFFF:08x}"


def _author() -> str:
    return getpass.getuser()


def _serialized_size_mb(save_fn) -> float:
    buffer = io.BytesIO()
    save_fn(buffer)
    return len(buffer.getvalue()) / (1024 * 1024)


def _plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.4, s=10)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted duration (min)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_feature_importance(
    feature_names: list[str], importances: np.ndarray, path: Path, top_n: int = 20
) -> None:
    order = np.argsort(np.abs(importances))[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(order))))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(order)} feature importances")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _requirements_txt() -> Path:
    tmp_path = Path(tempfile.mkstemp(suffix="_requirements.txt")[1])
    result = subprocess.run(
        ["pip", "freeze"], capture_output=True, text=True, check=True
    )
    tmp_path.write_text(result.stdout)
    return tmp_path


def _prepare_xy(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[DictVectorizer, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dicts = to_feature_dicts(df_train)
    test_dicts = to_feature_dicts(df_test)

    y_train = df_train[config.target].values.astype(np.float32)
    y_test = df_test[config.target].values.astype(np.float32)

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts).toarray().astype(np.float32)
    X_test = dv.transform(test_dicts).toarray().astype(np.float32)

    return dv, X_train, X_test, y_train, y_test


def _common_log(
    *,
    model: object,
    model_name: str,
    params: dict,
    rmse: float,
    mae: float,
    r2: float,
    train_seconds: float,
    model_size_mb: float,
    data_version: str,
    dv: DictVectorizer,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    importances: np.ndarray | None,
) -> None:
    mlflow.log_params(
        {**params, "split_seed": config.random_state, "data_version_hash": data_version}
    )
    mlflow.log_metrics(
        {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_duration_seconds": train_seconds,
            "model_size_mb": model_size_mb,
        }
    )
    mlflow.set_tags(
        {
            "git_commit": _git_sha(),
            "data_version": data_version,
            "author": _author(),
            "framework": model_name,
        }
    )

    n_features = len(feature_names)
    onnx_model = _to_onnx(model, model_type=model_name, n_features=n_features)
    mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path="model")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        residual_path = tmp_dir_path / "residuals.png"
        _plot_residuals(y_test, y_pred, residual_path)
        mlflow.log_artifact(str(residual_path))

        if importances is not None:
            importance_path = tmp_dir_path / "feature_importance.png"
            _plot_feature_importance(feature_names, importances, importance_path)
            mlflow.log_artifact(str(importance_path))

        dv_path = tmp_dir_path / "dv.pkl"
        with open(dv_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(dv_path))

        pkl_model_path = tmp_dir_path / "model_baseline.pkl"
        with open(pkl_model_path, "wb") as f_out:
            pickle.dump(model, f_out)
        mlflow.log_artifact(str(pkl_model_path))

    requirements_path = _requirements_txt()
    try:
        mlflow.log_artifact(str(requirements_path))
    finally:
        requirements_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Linear Regression — always-on baseline
# --------------------------------------------------------------------------- #
def train_linear_regression(
    df_train: pd.DataFrame, df_test: pd.DataFrame, run_name: str | None = None
) -> dict:
    dv, X_train, X_test, y_train, y_test = _prepare_xy(df_train, df_test)
    data_version = _data_version_hash(df_train, df_test)
    params = {"fit_intercept": True}

    with mlflow.start_run(run_name=run_name or "linear-regression-baseline"):
        start = time.perf_counter()
        model = LinearRegression(**params)
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_size_mb = _serialized_size_mb(lambda buf: pickle.dump(model, buf))

        mlflow.sklearn.log_model(model, name="model")

        _common_log(
            model=model,
            model_name="sklearn",
            params=params,
            rmse=rmse,
            mae=mae,
            r2=r2,
            train_seconds=train_seconds,
            model_size_mb=model_size_mb,
            data_version=data_version,
            dv=dv,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=dv.get_feature_names_out().tolist(),
            importances=model.coef_,
        )

        logger.info("linear_regression_trained", rmse=rmse, mae=mae, r2=r2)
        return {"rmse": rmse, "mae": mae, "r2": r2}


# --------------------------------------------------------------------------- #
# XGBoost — autologging enabled here
# --------------------------------------------------------------------------- #
def train_xgboost(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    params: dict | None = None,
    run_name: str | None = None,
    nested: bool = False,
) -> dict:
    mlflow.xgboost.autolog(log_models=True, log_input_examples=False)

    dv, X_train, X_test, y_train, y_test = _prepare_xy(df_train, df_test)
    data_version = _data_version_hash(df_train, df_test)
    params = params or {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": config.random_state,
    }

    with mlflow.start_run(run_name=run_name or "xgboost-baseline", nested=nested):
        start = time.perf_counter()
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        train_seconds = time.perf_counter() - start

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_size_mb = _serialized_size_mb(lambda buf: pickle.dump(model, buf))

        _common_log(
            model=model,
            model_name="xgboost",
            params=params,
            rmse=rmse,
            mae=mae,
            r2=r2,
            train_seconds=train_seconds,
            model_size_mb=model_size_mb,
            data_version=data_version,
            dv=dv,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=dv.get_feature_names_out().tolist(),
            importances=model.feature_importances_,
        )

        logger.info("xgboost_trained", rmse=rmse, mae=mae, r2=r2)
        return {"rmse": rmse, "mae": mae, "r2": r2}


# --------------------------------------------------------------------------- #
# PyTorch MLP
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(
        self, n_features: int, hidden_sizes: tuple[int, ...] = (64, 32)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = n_features
        for hidden in hidden_sizes:
            layers += [nn.Linear(in_features, hidden), nn.ReLU()]
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _TorchRegressorAdapter(BaseEstimator, RegressorMixin):
    def __init__(self, model: "_MLP" = None) -> None:
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TorchRegressorAdapter":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.as_tensor(X, dtype=torch.float32)).numpy()


def train_pytorch_mlp(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    params: dict | None = None,
    run_name: str | None = None,
) -> dict:
    dv, X_train, X_test, y_train, y_test = _prepare_xy(df_train, df_test)
    data_version = _data_version_hash(df_train, df_test)
    params = params or {
        "hidden_sizes": (64, 32),
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 16,
    }

    torch.manual_seed(config.random_state)

    with mlflow.start_run(run_name=run_name or "pytorch-mlp-baseline"):
        model = _MLP(n_features=X_train.shape[1], hidden_sizes=params["hidden_sizes"])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_fn = nn.MSELoss()

        X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32)

        start = time.perf_counter()
        model.train()
        n = X_train_t.shape[0]
        batch_size = min(params["batch_size"], n)
        for epoch in range(params["epochs"]):
            permutation = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = permutation[i : i + batch_size]
                optimizer.zero_grad()
                preds = model(X_train_t[idx])
                loss = loss_fn(preds, y_train_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            mlflow.log_metric("train_mse_loss", epoch_loss / n, step=epoch)
        train_seconds = time.perf_counter() - start

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.as_tensor(X_test, dtype=torch.float32)).numpy()

        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_size_mb = _serialized_size_mb(
            lambda buf: torch.save(model.state_dict(), buf)
        )

        adapter = _TorchRegressorAdapter(model)
        perm_result = permutation_importance(
            adapter,
            X_test,
            y_test,
            scoring="neg_mean_squared_error",
            n_repeats=5,
            random_state=config.random_state,
        )

        _common_log(
            model=model,
            model_name="pytorch",
            params=params,
            rmse=rmse,
            mae=mae,
            r2=r2,
            train_seconds=train_seconds,
            model_size_mb=model_size_mb,
            data_version=data_version,
            dv=dv,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=dv.get_feature_names_out().tolist(),
            importances=perm_result.importances_mean,
        )

        logger.info("pytorch_mlp_trained", rmse=rmse, mae=mae, r2=r2)
        return {"rmse": rmse, "mae": mae, "r2": r2}


# --------------------------------------------------------------------------- #
# Hyperparameter sweep — Optuna, XGBoost, nested runs
# --------------------------------------------------------------------------- #
def run_xgboost_sweep(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_trials: int = 10,
    run_name: str | None = None,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": config.random_state,
        }
        metrics = train_xgboost(
            df_train,
            df_test,
            params=params,
            run_name=f"trial-{trial.number}",
            nested=True,
        )
        return metrics["rmse"]

    with mlflow.start_run(run_name=run_name or "xgboost-hpo-sweep"):
        study = optuna.create_study(direction="minimize", study_name="xgboost-sweep")
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_rmse", study.best_value)
        mlflow.set_tag("sweep_n_trials", str(n_trials))

    logger.info(
        "xgboost_sweep_complete",
        best_rmse=study.best_value,
        best_params=study.best_params,
    )
    return study


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_all_baselines(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    return {
        "linear_regression": train_linear_regression(df_train, df_test),
        "xgboost": train_xgboost(df_train, df_test),
        "pytorch_mlp": train_pytorch_mlp(df_train, df_test),
    }


def main() -> None:
    from prodml.data import load_and_split_data
    from prodml.features import feature_engineering

    setup_mlflow()

    df_train_raw, df_test_raw = load_and_split_data()
    df_train, df_test = feature_engineering(df_train_raw, df_test_raw)

    baseline_results = train_all_baselines(df_train, df_test)
    logger.info("baseline_training_complete", results=baseline_results)

    study = run_xgboost_sweep(df_train, df_test, n_trials=10)
    logger.info(
        "sweep_complete", best_params=study.best_params, best_rmse=study.best_value
    )


if __name__ == "__main__":
    main()
