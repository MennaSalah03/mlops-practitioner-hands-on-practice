"""
training script.

This is a dev-time script that runs once(or whenever you want to retrain)
to produce the model artifact that
the API loads at runtime.

Usage:
    python training/train.py
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"


def load_data():
    """Load the built-in sklearn diabetes dataset (regression target)."""
    dataset = load_diabetes(return_X_y= False, as_frame=True)
    X = dataset.data
    y = dataset.target
    feature_names = list(X.columns)
    logger.info("Loaded dataset: %d rows, %d features", len(X), len(feature_names))
    return X, y, feature_names


def train_model(X, y):
    """ Training linear regression model  """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    logger.info("Evaluation metrics: %s", metrics)

    return model, metrics


def save_artifact(model, feature_names, metrics):
    """ Save the model as a joblib artifact """
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "1.0.0",
    }

    joblib.dump(artifact, ARTIFACT_PATH)
    logger.info("Saved model artifact to %s", ARTIFACT_PATH)


def main():
    X, y, feature_names = load_data()
    model, metrics = train_model(X, y)
    save_artifact(model, feature_names, metrics)


if __name__ == "__main__":
    main()
