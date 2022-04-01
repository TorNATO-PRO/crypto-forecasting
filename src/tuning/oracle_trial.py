import random
from typing import Dict

import numpy as np

import nni
import torch

from ..data_loader.load import DataLoader, CryptoDataset
from ..models.oracle.oracle import train_model

# set seed
seed = 80085
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def prepare_oracle(params: Dict) -> np.float32:
    data = DataLoader()
    dataset = data.load_data(CryptoDataset.BITCOIN)
    val_min, _ = train_model(dataset, '2017-01-01', '2021-01-01', params)
    return val_min


val = prepare_oracle(nni.get_next_parameter())
nni.report_final_result(val)
