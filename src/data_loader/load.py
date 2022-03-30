"""
Author: Nathan Waltz

Implemented loading utilities for the datasets.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from enum import Enum


class CryptoDataset(Enum):
    """
    An enum that maps the name to a string.
    """
    BITCOIN = 'BTC-USD.csv'
    ETHEREUM = 'ETH-USD.csv'


class DataLoader:
    """
    Helper class for loading data from a variety of sources.
    """

    def __init__(self):
        """
        Constructs a new instance of the DataLoader class.
        """
        self._data_path = (Path(__file__).parents[2]).joinpath('assets').joinpath('datasets')
        self.data_type_dict = {'Close': np.float32}

    def load_data(self, dataset: CryptoDataset) -> pd.DataFrame:
        """
        Loads a typed cryptocurrency dataset.

        :param dataset: The type of CryptoDataset to load.
        :return: A dataframe that corresponds to that dataset.
        """
        path = self._data_path.joinpath(dataset.value)
        df: pd.DataFrame = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        df.astype({col: np.float32 for col in df.columns})
        return df
