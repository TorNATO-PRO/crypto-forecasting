import random
import numpy as np
import torch

from src.data_loader.load import DataLoader
from src.data_loader.load import CryptoDataset
from src.models.custom import custom
import src.models.oracle.oracle as oracle

# define the model parameters
params_oracle = {
    "lr": 0.005,
    "rnn_type": "gru",
    "rnn_hidden_size": 8,
    "ind_hidden_size": 2,
    "des_size": 8,
    "ind1": {
        "_name": "ao",
        'fast': 9,
        'slow': 14
    },
    "ind2": {
        "_name": "rsi",
        'length': 20
    }
}

params_custom = {
    "lr": 0.005,
    'rnn_hidden_size': 8,
    "rnn_agg_hidden_size": 8,
    "trading_ind_hidden_size": 2,
    "linear_agg_hidden_size": 2,
    "ind1": {
        "_name": "ao",
        'fast': 9,
        'slow': 14
    },
    "ind2": {
        "_name": "rsi",
        'length': 20
    }
}

# set seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data_loader = DataLoader()
dataset = data_loader.load_data(CryptoDataset.BITCOIN)
print(dataset.columns)
_, oracle_model = custom.train_model(dataset, '2017-01-01', '2021-01-01', params_custom)
custom.evaluate(dataset, '2021-01-01', '2022-01-01', params_custom, oracle_model)
