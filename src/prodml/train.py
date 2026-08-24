"""Fit the model and persist it (bundled with its fitted DictVectorizer)."""

import pickle
from pathlib import Path

import structlog
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import config
from prodml.data import load_and_split_data
from prodml.features import feature_engineering, to_feature_dicts

logger = structlog.get_logger()


def train_model(df_train, df_test) -> tuple[dict, dict]:
    """calls the training functions in features.py"""
    train_dicts = to_feature_dicts(df_train)
    test_dicts = to_feature_dicts(df_test)

    # Extract y BEFORE vectorizing X — don't reuse the variable names
    y_train = df_train[config.target].values
    y_test = df_test[config.target].values

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    logger.info("model_evaluated", rmse=rmse, mae=mae)

    artifact = {"dv": dv, "model": model}
    return artifact, {"rmse": rmse, "mae": mae}


def persist_model(artifact: dict, model_path: str | None = None) -> None:
    """saving the model as pickle file"""
    path = Path(model_path or config.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        pickle.dump(artifact, f_out)
    logger.info("model is saved", path=str(path))


def main() -> dict:
    df_train_raw, df_test_raw = load_and_split_data()
    df_train, df_test = feature_engineering(df_train_raw, df_test_raw)
    artifact, metrics = train_model(df_train, df_test)
    persist_model(artifact)
    return metrics


if __name__ == "__main__":
    main()
