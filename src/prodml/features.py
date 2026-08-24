"""Feature engineering — shared between training and inference to avoid skew."""

import pandas as pd

from prodml.config import config


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """computing the target - duration"""
    df = df.copy()
    df[config.target] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df[config.target] >= config.min_duration)
        & (df[config.target] <= config.max_duration)
    ].copy()


def add_pu_do_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_col = config.categorical[0]
    df[cat_col] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: target -> outlier filter -> PU_DO."""
    df = compute_target(df)
    df = filter_outliers(df)
    df = add_pu_do_feature(df)
    return df


def feature_engineering(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return prepare_features(df_train), prepare_features(df_test)


def to_feature_dicts(df: pd.DataFrame) -> list[dict]:
    """Selects the model's input columns as records, ready for DictVectorizer."""
    return df[config.categorical + config.numerical].to_dict(orient="records")


def prepare_inference_record(record: dict) -> dict:
    """
    Applies the SAME PU_DO derivation used at training time to one raw
    inference record (e.g. from an API request). This is the single source
    of truth that keeps train and serve consistent.
    """
    return {
        config.categorical[0]: f"{record['PULocationID']}_{record['DOLocationID']}",
        config.numerical[0]: record["trip_distance"],
    }
