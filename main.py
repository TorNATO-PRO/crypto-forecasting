import random
import numpy as np
import torch

from src.data_loader.load import DataLoader
from src.data_loader.load import CryptoDataset
import src.models.oracle.oracle as oracle

# define the model parameters
params = {
    "lr": 0.01,
    "rnn_type": "rnn",
    "rnn_hidden_size": 24,
    "ind_hidden_size": 4,
    "des_size": 8,
    "ind1": {
        "_name": "apo",
        'fast': 3,
        'slow': 40
    },
    "ind2": {
        "_name": "cci",
        'length': 5
    }
}

# set seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data = DataLoader()
dataset = data.load_data(CryptoDataset.BITCOIN)
_, oracle_model = oracle.train_model(dataset, '2017-01-01', '2021-01-01', params, save_model=True)
oracle.evaluate(dataset, '2021-01-01', '2022-01-01', params, oracle_model)
