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
    "rnn_type": "rnn",
    "rnn_hidden_size": 24,
    "ind_hidden_size": 2,
    "des_size": 16,
    "ind1": {
        "_name": "mom",
        "length": 5
    },
    "ind2": {
        "_name": "rsi",
        "length": 20
    }
}

params_custom = {
    "lr": 0.01,
    "rnn_hidden_size": 32,
    "rnn_agg_hidden_size": 32,
    "trading_ind_hidden_size": 16,
    "linear_agg_hidden_size": 4,
    "ind1": {
        "_name": "cci",
        "length": 20
    },
    "ind2": {
        "_name": "cci",
        "length": 20
    }
}

# set seed
seed = 80085
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data_loader = DataLoader()
dataset = data_loader.load_data(CryptoDataset.ETHEREUM)
_, custom_model = custom.train_model(dataset, '2018-01-01', '2021-01-01', params_custom, ['Open'])
custom.evaluate(dataset, '2021-01-01', '2022-01-01', params_custom, ['Open'], custom_model)

# set seed
seed = 80085
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

_, oracle_model = oracle.train_model(dataset, '2018-01-01', '2021-01-01', params_oracle)
oracle.evaluate(dataset, '2021-01-01', '2022-01-01', params_oracle, oracle_model)
