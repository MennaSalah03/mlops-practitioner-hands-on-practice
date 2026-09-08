"""Unit tests for prodml.predict.DurationPredictor (onnxruntime-backed).

Uses exported_models (conftest.py) to get real .onnx + .pkl files on disk,
and the predictor fixture for an already-loaded instance. No mocking of
onnxruntime or sklearn — real artifacts, real inference.
"""

import onnxruntime as ort
import pytest

from prodml.predict import DurationPredictor


# --------------------------------------------------------------------------- #
# load()
# --------------------------------------------------------------------------- #
def test_load_returns_self(exported_models):
    result = DurationPredictor().load()
    assert isinstance(result, DurationPredictor)


def test_load_sets_dv_and_session(predictor):
    assert predictor._dv is not None
    assert predictor._session is not None
    assert predictor._input_name is not None


def test_load_missing_onnx_file_raises(exported_models, tmp_path):
    missing_onnx = str(tmp_path / "does_not_exist.onnx")
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.NoSuchFile):
        DurationPredictor().load(model_path=missing_onnx)


def test_load_missing_pickle_file_raises(exported_models, tmp_path):
    missing_pkl = str(tmp_path / "does_not_exist.pkl")
    with pytest.raises(FileNotFoundError):
        DurationPredictor().load(dv_path=missing_pkl)


# --------------------------------------------------------------------------- #
# guard: predict before load
# --------------------------------------------------------------------------- #
def test_predict_one_before_load_raises(sample_inference_record):
    predictor = DurationPredictor()
    with pytest.raises(RuntimeError, match="not loaded"):
        predictor.predict_one(sample_inference_record)


def test_predict_batch_before_load_raises(sample_inference_record):
    predictor = DurationPredictor()
    with pytest.raises(RuntimeError, match="not loaded"):
        predictor.predict_batch([sample_inference_record])


# --------------------------------------------------------------------------- #
# predict_one()
# --------------------------------------------------------------------------- #
def test_predict_one_matches_sklearn_baseline(
    predictor, fitted_artifact, sample_inference_record
):
    """Ground truth: run the same request through the pickled dv + sklearn
    model directly, bypassing onnxruntime entirely, and compare."""
    result = predictor.predict_one(sample_inference_record)

    dv = fitted_artifact["dv"]
    model = fitted_artifact["model"]
    feature_dict = {
        "PU_DO": f"{sample_inference_record['PULocationID']}_{sample_inference_record['DOLocationID']}",
        "trip_distance": sample_inference_record["trip_distance"],
    }
    expected = float(model.predict(dv.transform([feature_dict]))[0])

    assert result == pytest.approx(expected, rel=1e-3)


def test_predict_one_returns_python_float(predictor, sample_inference_record):
    result = predictor.predict_one(sample_inference_record)
    assert isinstance(result, float)  # not np.float32 — matters for JSON serialization


def test_predict_one_unseen_pu_do_pair_does_not_crash(predictor):
    """DictVectorizer silently drops unseen keys/values at transform time —
    a PU_DO pair never seen in training just contributes zero, no error."""
    unseen_record = {"PULocationID": 999, "DOLocationID": 999, "trip_distance": 5.0}
    result = predictor.predict_one(unseen_record)
    assert isinstance(result, float)


def test_predict_one_missing_raw_field_raises_keyerror(predictor):
    incomplete_record = {
        "PULocationID": 100,
        "trip_distance": 5.0,
    }  # missing DOLocationID
    with pytest.raises(KeyError):
        predictor.predict_one(incomplete_record)


# --------------------------------------------------------------------------- #
# predict_batch()
# --------------------------------------------------------------------------- #
def test_predict_batch_returns_list_of_floats(predictor, sample_inference_record):
    records = [
        sample_inference_record,
        {"PULocationID": 101, "DOLocationID": 132, "trip_distance": 12.0},
    ]
    results = predictor.predict_batch(records)
    assert isinstance(results, list)
    assert len(results) == len(records)
    assert all(isinstance(r, float) for r in results)


def test_predict_batch_matches_predict_one_elementwise(
    predictor, sample_inference_record
):
    """Catches ordering/shape bugs that only show up with >1 row."""
    records = [
        sample_inference_record,
        {"PULocationID": 101, "DOLocationID": 132, "trip_distance": 12.0},
        {"PULocationID": 100, "DOLocationID": 18, "trip_distance": 8.0},
    ]

    batch_results = predictor.predict_batch(records)
    individual_results = [predictor.predict_one(r) for r in records]

    for batch_val, individual_val in zip(batch_results, individual_results):
        assert batch_val == pytest.approx(individual_val, rel=1e-3)


def test_predict_batch_empty_list(predictor):
    assert predictor.predict_batch([]) == []
