import random
import numpy as np
import torch

from src.data_loader.load import DataLoader
from src.data_loader.load import CryptoDataset
import src.models.oracle.oracle as oracle

# define the model parameters
params = {
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

# set seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data = DataLoader()
dataset = data.load_data(CryptoDataset.BITCOIN)
_, oracle_model = oracle.train_model(dataset, '2017-01-01', '2021-01-01', params)
oracle.evaluate(dataset, '2021-01-01', '2022-01-01', params, oracle_model)
