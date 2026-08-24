"""Loading and splitting data for training."""

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from prodml.config import config

logger = structlog.get_logger()


def load_and_split_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the raw Parquet file and splits it using centralized config."""
    logger.info("loading_data", path=config.data_path)
    df = pd.read_parquet(config.data_path)

    df_train, df_test = train_test_split(
        df, test_size=config.test_size, random_state=config.random_state
    )
    logger.info("split_data", train_rows=len(df_train), test_rows=len(df_test))
    return df_train, df_test
