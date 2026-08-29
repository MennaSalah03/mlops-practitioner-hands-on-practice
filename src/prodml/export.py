import json
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

    dv = artifact["dv"]
    model = artifact["model"]
    metadata = artifact["metadata"]

    pipeline = Pipeline(
        [
            ("vectorizer", dv),
            (
                "regressor",
                model,
            ),
        ]
    )

    initial_type = [
        ("input_dict", DictionaryType(StringTensorType([1]), FloatTensorType([1])))
    ]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)

    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value) if not isinstance(value, str) else value

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        f_out.write(onnx_model.SerializeToString())

    with open(config.metadata_path, "w") as f_out:
        json.dump(metadata, f_out, indent=2)

    logger.info(
        "model is saved",
        path=str(path),
        metadata_path=str(config.metadata_path),
        artifact_hash=metadata["artifact_hash"],
        model_version=metadata["model_version"],
    )


def main() -> None:
    df_train_raw, df_test_raw = load_and_split_data()
    df_train, df_test = feature_engineering(df_train_raw, df_test_raw)
    artifact = train_model(df_train, df_test)
    persist_model(artifact)
    # return metrics


if __name__ == "__main__":
    main()
