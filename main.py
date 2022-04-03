from datetime import datetime
import random
import numpy as np
import torch

from src.data_loader.load import DataLoader
from src.data_loader.load import CryptoDataset
from src.models.custom import custom
import src.models.oracle.oracle as oracle
from matplotlib import pyplot as plt
import pandas as pd
from tabulate import tabulate

# define the model parameters
params_oracle = {
    "lr": 0.01,
    "rnn_type": "gru",
    "rnn_hidden_size": 24,
    "ind_hidden_size": 2,
    "des_size": 8,
    "ind1": {
        "_name": "cmo",
        "length": 5
    },
    "ind2": {
        "_name": "cci",
        "length": 5
    }
}

params_custom = {
    "lr": 0.1,
    "rnn_hidden_size": 12,
    "rnn_agg_hidden_size": 8,
    "trading_ind_hidden_size": 16,
    "linear_agg_hidden_size": 16,
    "ind1": {
        "_name": "cci",
        "length": 5
    },
    "ind2": {
        "_name": "rsi",
        "length": 5
    }
}

# set seed
seed = 80085
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# dates
start_date = '2018-01-01'
end_date = '2021-01-01'
start_pred_date = '2021-01-01'
end_pred_date = '2022-04-01'


data_loader = DataLoader()
crypto_dataset = CryptoDataset('BITCOIN', 'BTC-USD.csv')
dataset = data_loader.load_data(crypto_dataset)


header = '#########################################'
oracle_msg = 'Learning using Oracle'
oracle_pad_len = (len(header) - len(oracle_msg) - 2) // 2
custom_msg = 'Learning using Custom Model'
custom_pad_len = (len(header) - len(custom_msg) - 2) // 2
summary_msg = 'Learning using Summary'
summary_pad_len = (len(header) - len(summary_msg) - 2) // 2


print("#########################################")
print('#' + oracle_pad_len * ' ' + oracle_msg +  oracle_pad_len * ' ' + '#')
print("#########################################")
_, oracle_model = oracle.train_model(dataset, start_date, end_date, params_oracle)
ora_preds, buy_hold_preds, ora_trades = oracle.evaluate(dataset, start_pred_date, end_pred_date, params_oracle, oracle_model)

d = [[f'Day {day}', action] for day, action in enumerate(ora_trades)]
df = pd.DataFrame(d, columns = ['Day of Trading', 'Shares Bought'])
print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

print("#########################################")
print('#' + custom_pad_len * ' ' + custom_msg +  custom_pad_len * ' ' + '#')
print("#########################################")
_, custom_model = custom.train_model(dataset, start_date, end_date, params_custom, ['Open'])
cus_preds, cus_trades = custom.evaluate(dataset, start_pred_date, end_pred_date, params_custom, ['Open'], custom_model)

d = [[f'Day {day}', action] for day, action in enumerate(cus_trades)]
df = pd.DataFrame(d, columns = ['Day of Trading', 'Shares Bought'])
print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

print("#########################################")
print('#' + summary_pad_len * ' ' + summary_msg +  summary_pad_len * ' ' + '#')
print("#########################################")
d = [ ["Buy and Hold", round(buy_hold_preds[-1], 4), np.mean(buy_hold_preds)],
     ["Oracle", round(ora_preds[-1], 4), np.mean(ora_preds)],
     ["Custom", round(cus_preds[-1], 4), np.mean(cus_preds)]]

df = pd.DataFrame(d, columns = ['Model','Final Return','Mean Return'])
print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

print("Press any key to show plot")
input()

# sanity check
assert(len(ora_preds) == len(buy_hold_preds) == len(cus_preds))

# plotting logic
plt.title(f'Trading evaluation from {start_pred_date} to {end_pred_date} on {str(crypto_dataset)}')
plt.plot(ora_preds, label='Oracle Returns')
plt.plot(buy_hold_preds, label='Buy and Hold Returns')
plt.plot(cus_preds, label='Model Returns')
plt.xlabel('Day of Trading')
plt.ylabel('Model Returns (USD)')
plt.xticks(rotation=45)
plt.axhline(y=0, color='black', linestyle='--')
plt.legend()
plt.show()
