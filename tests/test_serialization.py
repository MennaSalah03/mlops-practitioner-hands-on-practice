"""ONNX vs pickle parity test — same fitted model, two serialization formats,
predictions must agree. Uses the pickle baseline as ground truth since it's
the direct sklearn output with nothing lost in translation.
"""

import pickle

import numpy as np
import onnxruntime as ort


def test_onnx_pickle_parity(exported_models, sample_feature_dicts):
    """Ensures ONNX and Pickle pipelines produce identical predictions."""
    # Load Pickle baseline
    with open(exported_models["pkl"], "rb") as f:
        artifact = pickle.load(f)
    dv = artifact["dv"]
    model = artifact["model"]

    # Predict Pickle (Baseline)
    X_matrix = dv.transform(sample_feature_dicts)
    pkl_preds = model.predict(X_matrix)

    # Predict ONNX — export.py's persist_model_onnx converts only the
    # regression model, not the DictVectorizer, so the ONNX graph expects
    # the same dense float matrix dv.transform() produces, not a raw dict.
    # dv.transform() returns a scipy sparse matrix; onnxruntime needs dense.
    session = ort.InferenceSession(exported_models["onnx"])
    input_name = session.get_inputs()[0].name
    X_onnx = np.asarray(X_matrix.todense(), dtype=np.float32)
    onnx_preds = session.run(None, {input_name: X_onnx})[0].flatten()

    # Assert Parity
    np.testing.assert_allclose(
        pkl_preds,
        onnx_preds,
        rtol=1e-4,
        atol=1e-4,
        err_msg="ONNX predictions diverged from Pickle baseline.",
    )
