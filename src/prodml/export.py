import json
import pickle
from pathlib import Path

import structlog
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import config
from prodml.data import load_and_split_data
from prodml.features import feature_engineering
from prodml.train import train_model

logger = structlog.get_logger()


def persist_model_onnx(artifact: dict, model_path: str | None = None) -> None:
    """Saving only the regression model as ONNX — not the DictVectorizer."""
    path = Path(model_path or config.onnx_model_path)

    model = artifact["model"]
    metadata = artifact["metadata"]
    n_features = len(metadata["feature_names"])
    n_features = model.n_features_in_

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

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


def persist_model_pickle(artifact: dict, model_path: str | None = None) -> None:
    """Saves the complete training artifact (dv + model + metadata) as pickle."""
    path = Path(model_path or config.pickle_model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        pickle.dump(artifact, f_out)
    logger.info(
        "model_pickled_successfully",
        path=str(path),
        artifact_hash=artifact["metadata"]["artifact_hash"],
        model_version=artifact["metadata"]["model_version"],
    )


def main() -> None:
    df_train_raw, df_test_raw = load_and_split_data()
    df_train, df_test = feature_engineering(df_train_raw, df_test_raw)
    artifact = train_model(df_train, df_test)
    persist_model_pickle(artifact)
    persist_model_onnx(artifact)


if __name__ == "__main__":
    main()
