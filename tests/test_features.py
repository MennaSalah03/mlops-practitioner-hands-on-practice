from prodml.features import (
    add_pu_do_feature,
    compute_target,
    filter_outliers,
    prepare_inference_record,
)


def test_compute_target(sample_raw_dataframe, mock_config):
    df = compute_target(sample_raw_dataframe)

    assert mock_config.target in df.columns
    # 15 mins, 0 mins, 100 mins, 10 mins
    expected_durations = [15.0, 0.0, 100.0, 10.0]
    assert df[mock_config.target].tolist() == expected_durations


def test_filter_outliers(sample_raw_dataframe, mock_config):
    df = compute_target(sample_raw_dataframe)
    df_filtered = filter_outliers(df)

    # Should drop the 0 min and 100 min trips based on mock_config (1 to 60)
    assert len(df_filtered) == 2
    assert df_filtered[mock_config.target].min() >= mock_config.min_duration
    assert df_filtered[mock_config.target].max() <= mock_config.max_duration


def test_add_pu_do_feature_including_nans(sample_raw_dataframe, mock_config):
    df = add_pu_do_feature(sample_raw_dataframe)

    assert mock_config.categorical[0] in df.columns
    pu_do_values = df[mock_config.categorical[0]].tolist()

    assert pu_do_values[0] == "100_10"
    assert pu_do_values[1] == "101_132"
    # Ensure pandas astype(str) conversion on NaNs results in explicit string fallback
    assert pu_do_values[3] == "nan_nan"


def test_prepare_inference_record_zero_distance_and_nans(mock_config):
    # Edge case: zero distance and missing IDs via raw dict
    raw_record = {
        "PULocationID": float("nan"),
        "DOLocationID": float("nan"),
        "trip_distance": 0.0,
    }

    feature_dict = prepare_inference_record(raw_record)

    assert feature_dict[mock_config.categorical[0]] == "nan_nan"
    assert feature_dict[mock_config.numerical[0]] == 0.0
