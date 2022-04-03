import random
from typing import Dict

import numpy as np

import nni
import torch

from ..data_loader.load import DataLoader, CryptoDataset
from ..models.custom.custom import train_model

# set seed
seed = 80085
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def prepare_custom(params: Dict) -> float:
    """
    Helper function for AutoML, helps to tune our hyper-parameters.

    Args:

    :param params: The hyper-parameters to test.
    """
    data = DataLoader()
    dataset = data.load_data(CryptoDataset("BITCOIN", "BTC-USD.csv"))
    val_min, _ = train_model(
        dataset, "2017-01-01", "2021-01-01", params, ["Open", "Close"]
    )
    return val_min


val = prepare_custom(nni.get_next_parameter())
nni.report_final_result(val)
