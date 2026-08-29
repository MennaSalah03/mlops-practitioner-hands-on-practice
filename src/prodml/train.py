"""Fit the model and persist it (bundled with its fitted DictVectorizer)."""

import structlog
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import config
from prodml.features import to_feature_dicts
from prodml.metadata import build_metadata

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

    payload = {"dv": dv, "model": model}
    metrics = {"rmse": rmse, "mae": mae}
    metadata = build_metadata(payload, metrics)

    artifact = {**payload, "metadata": metadata.to_dict()}
    return artifact
