from pathlib import Path

import structlog
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    DictionaryType,
    FloatTensorType,
    StringTensorType,
)
from sklearn.pipeline import Pipeline

from prodml.config import config
from prodml.data import load_and_split_data
from prodml.features import feature_engineering
from prodml.train import train_model

logger = structlog.get_logger()


def persist_model(artifact: dict, model_path: str | None = None) -> None:
    """saving the model as an onnx file"""
    path = Path(model_path or config.model_path)

    dv = artifact.get("dv")
    model = artifact.get("model")

    pipeline = Pipeline(
        [
            ("vectorizer", dv),
            (
                "classifier",
                model,
            ),  # Adjust name to 'regressor' if this is a regression model
        ]
    )

    initial_type = [
        ("input_dict", DictionaryType(StringTensorType([1]), FloatTensorType([1])))
    ]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        f_out.write(onnx_model.SerializeToString())

    logger.info("model is saved", path=str(path))


def main() -> dict:
    df_train_raw, df_test_raw = load_and_split_data()
    df_train, df_test = feature_engineering(df_train_raw, df_test_raw)
    artifact, metrics = train_model(df_train, df_test)
    persist_model(artifact)
    return metrics


if __name__ == "__main__":
    main()
