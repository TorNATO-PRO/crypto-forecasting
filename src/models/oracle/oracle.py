"""
An Oracle for the crypto problem. Using a "stock market prediction" sample
found in Ivan Gridin's book "Time Series Forecasting using Deep Learning:
Combining Pytorch, RNN, TCN, and Deep Neural Network Models to Provide
Production-Ready Prediction Solutions." with ISBN-13: 978-9391392574.

The codebase has an MIT license, so I am assuming it is safe to use:
https://github.com/bpbpublications/Time-Series-Forecasting-using-Deep-Learning/tree/main/Chapter%2007/stock
"""
import os
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd

# trust me, we need this for the trading indicators
import pandas_ta as ta
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from collections import OrderedDict

from scipy.stats import norm
from torch.optim import Adam
from src.models.baseline.baseline import buy_and_hold
from src.models.loss import NegativeMeanReturnLoss

# check whether it can run on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class Oracle(nn.Module):
    """
    The Oracle, adapted from the Ivan Gridin's solution in 978-9391392574.
    """

    def __init__(self,
                 rnn_input_size: int,
                 ind_input_size: int,
                 rnn_type: str = 'gru',
                 rnn_hidden_size: int = 16,
                 ind_hidden_size: int = 4,
                 des_size: int = 4):
        """
        Constructs a new instance of the Oracle class.
        """
        super(Oracle, self).__init__()
        self.oracle_model = 'Ivan\'s Alg Trader'
        rnn_params = {
            'input_size': rnn_input_size,
            'hidden_size': rnn_hidden_size,
            'batch_first': True
        }

        if rnn_type == 'gru':
            self.rnn = nn.GRU(**rnn_params)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(**rnn_params)
        else:
            raise Exception(f'This type of RNN is not supported: {rnn_type}')

        self.lin_ind = nn.Linear(ind_input_size, ind_hidden_size)
        self.lin_des = nn.Linear(rnn_hidden_size + ind_hidden_size, des_size)
        self.lin_pos = nn.Linear(des_size, 1)

    def forward(self, raw: torch.Tensor, indicators: torch.Tensor) -> torch.Tensor:
        """
        The forward method, defines how the model
        shall be run.

        :param raw: The raw data.
        :param indicators: Trade indicators
        :return: An output tensor which results after
        the input data goes through the model.
        """
        _, h = self.rnn(raw)
        z = torch.relu(self.lin_ind(indicators))
        x = torch.cat((z, h[0]), dim=1)
        x = torch.relu(self.lin_des(x))
        p = torch.tanh(self.lin_pos(x))
        return p.view(-1)


def sliding_window(time_series: List[List[np.float32]],
                   num_features: int,
                   target_len: int = 1) -> Tuple[List[List[List[np.float32]]], List[List[List[np.float32]]]]:
    """
    The sliding window for a given time series. Slices the passed
    time_series up into windows with length num_features and target length
    target_len.

    :param time_series: The time series to slice up.
    :param num_features: The length of time windows to use.
    :param target_len: The length of the target, or Y.
    :return: Time windows for X and Y.
    """
    x = [time_series[i:i + num_features] for i in range(len(time_series) + 1 - num_features - target_len)]
    y = [time_series[i:i + target_len] for i in range(num_features, len(time_series) + 1 - target_len)]
    return x, y


def get_indicator(dataframe: pd.DataFrame,
                  indicator_name: str,
                  parameters: Dict[str, Union[int, str]]) -> pd.DataFrame:
    """
    Gets a specified indicator with a given indicator name.

    :param dataframe: A pandas dataframe.
    :param indicator_name: The name of the indicator.
    :param parameters: Relevant parameters.
    :return: A new dataframe with the indicator operation performed on it.
    """
    if indicator_name == 'ao':
        return dataframe.ta.ao(parameters['fast'], parameters['slow'])
    elif indicator_name == 'apo':
        return dataframe.ta.apo(parameters['fast'], parameters['slow'])
    elif indicator_name == 'cci':
        return dataframe.ta.cci(parameters['length']) / 100
    elif indicator_name == 'cmo':
        return dataframe.ta.cmo(parameters['length']) / 100
    elif indicator_name == 'mom':
        return dataframe.ta.mom(parameters['length'])
    elif indicator_name == 'rsi':
        return dataframe.ta.rsi(parameters['length']) / 100

    raise Exception('That indicator is not supported')


def train_model(data: pd.DataFrame,
                start_date: str,
                end_date: str,
                parameters: Dict,
                window_size: int = 40,
                num_epochs: int = 500,
                train_val_ratio: float = 0.8) -> Tuple[np.float32, nn.Module]:
    """
    Trains the model.

    :param data: The data to use.
    :param start_date: The date to start with.
    :param end_date: The date to end with.
    :param window_size: The window size.
    :param num_epochs: The number of epochs to train the model for.
    :param train_val_ratio: How much we want for train and validation.
    :param parameters: Model hyperparameters.
    :return: The minimum validation loss and the model.
    """
    # cut after end date
    data = data[data.index < end_date]
    ts_len = data[data.index > start_date].shape[0]
    train_length = int(ts_len * train_val_ratio)

    # get the hyperparameters
    learning_rate = parameters['lr']
    rnn_type = parameters['rnn_type']
    rnn_hidden_size = parameters['rnn_hidden_size']
    ind_hidden_size = parameters['ind_hidden_size']
    des_size = parameters['des_size']
    ind1_name = parameters['ind1']['_name']
    ind2_name = parameters['ind2']['_name']

    # an ordered dictionary of data
    data_source = OrderedDict()
    data_source['close_diff'] = (data['Close'] - data['Close'].shift(1))
    data_source['close_roc'] = (data['Close'] / data['Close'].shift(1))
    data_source['ind1'] = get_indicator(data, ind1_name, parameters['ind1'])
    data_source['ind2'] = get_indicator(data, ind2_name, parameters['ind2'])

    # add value at risk as an indicator
    # pct_changes = data['Close'].pct_change()
    # value_at_risk = []
    # for i in range(len(pct_changes)):
    #     mean = pct_changes[:i].mean()
    #     std = pct_changes[:i].std()
    #     value_at_risk_pct = abs(norm.ppf(0.01, mean, std))
    #     value_at_risk.append(value_at_risk_pct)

    # data_source['ind3'] = pd.Series(value_at_risk)
    # data_source['ind3'].index = pct_changes.index

    # Cut to 'start date'
    for k, v in data_source.items():
        data_source[k] = v[v.index >= start_date].dropna().values

    # aggregate all of that data
    data_aggregation = [[v[i] for _, v in data_source.items()] for i in range(ts_len)]

    x, y = sliding_window(data_aggregation, window_size)
    x_train, y_train = x[:train_length], y[:train_length]
    x_val, y_val = x[train_length:], y[train_length:]

    x_train = torch.tensor(x_train).to(device).float()
    y_train = torch.tensor(y_train).to(device).float()
    x_val = torch.tensor(x_val).to(device).float()
    y_val = torch.tensor(y_val).to(device).float()

    # get separate inputs, outputs
    close_train, close_val = x_train[:, :, :2], x_val[:, :, :2]
    index_train, index_val = x_train[:, -1, 2:], x_val[:, -1, 2:]
    price_train, price_val = y_train[:, :, 0].view(-1), y_val[:, :, 0].view(-1)

    # Initialize model
    model_params = {
        'rnn_input_size': 2,
        'ind_input_size': 2,
        'rnn_type': rnn_type,
        'rnn_hidden_size': rnn_hidden_size,
        'ind_hidden_size': ind_hidden_size,
        'des_size': des_size
    }
    model = Oracle(**model_params).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = NegativeMeanReturnLoss()

    # train the model
    val_losses = []
    for e in range(num_epochs):
        model.train()
        predicted = model(close_train, index_train)
        loss = criterion(predicted, price_train)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_predicted = model(close_val, index_val)
            val_loss = criterion(val_predicted, price_val)
            val_losses.append(val_loss)

        if e % 10 == 0:
            print(f'Epoch {e} | train: {loss.item()}, '
                  f'val: {val_loss.item()}')

    return torch.tensor(val_losses).mean(), model


def evaluate(data: pd.DataFrame,
             start_date: str,
             end_date: str,
             parameters: Dict,
             model: nn.Module = None,
             window_size: int = 40) -> None:
    """
    Trains the model.

    :param model: The model that is being evaluated.
    :param start_date: The date to start with.
    :param end_date: The date to end with.
    :param data: The data that is being used to train the model.
    :param parameters: Model hyperparameters.
    :param window_size: The window size.
    :return: Nothing.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = data[data.index < end_date]
    ts_len = data[data.index > start_date].shape[0]

    # an ordered dictionary of data
    data_source = OrderedDict()
    data_source['close_diff'] = (data['Close'] - data['Close'].shift(1))
    data_source['close_roc'] = (data['Close'] / data['Close'].shift(1))
    # data_source['pct_change'] = data['Close'].pct_change()
    data_source['ind1'] = get_indicator(data, parameters['ind1']['_name'], parameters['ind1'])
    data_source['ind2'] = get_indicator(data, parameters['ind2']['_name'], parameters['ind2'])
    data_source['close_diff'][0] = 0
    data_source['close_roc'][0] = 1

    # Cut to 'start date'
    for k, v in data_source.items():
        data_source[k] = v[v.index >= start_date].dropna().values

    # aggregate all of that data
    data_aggregation = [[v[i] for _, v in data_source.items()] for i in range(ts_len)]

    x, y = sliding_window(data_aggregation, window_size)
    x = torch.tensor(x).to(device).float()
    y = torch.tensor(y).to(device).float()

    cost = x[:, :, :2]
    index = x[:, -1, 2:]
    tomorrow_price_diff = y[:, :, 0].view(-1)

    model_params = {
        'rnn_input_size': 2,
        'ind_input_size': 2,
        'rnn_type': parameters['rnn_type'],
        'rnn_hidden_size': parameters['rnn_hidden_size'],
        'ind_hidden_size': parameters['ind_hidden_size'],
        'des_size': parameters['des_size']
    }

    if model is None:
        model = Oracle(**model_params).to(device)
        model.load_state_dict(torch.load(f'{dir_path}/data/oracle_best.pth'))

    # evaluate the model
    model.eval()
    with torch.no_grad():
        trades = model(cost, index)
        # Rounded Trades
        trades = torch.round(trades * 100) / 100

        # Calculating Absolute Returns
        abs_return = torch.mul(trades, tomorrow_price_diff)
        cumsum_return = [0] + torch.cumsum(abs_return, dim=0) \
            .view(-1).tolist()
        # Buy and Hold Strategy Returns
        buy_and_hold_returns = buy_and_hold(tomorrow_price_diff)

        print(f'Model Returns: {round(cumsum_return[-1], 4)}')
        print(f'Model Mean returns: {np.mean(cumsum_return)}')
        print(f'Buy and Hold Returns: {round(buy_and_hold_returns[-1], 4)}')
        print(f'Buy and Hold Mean Returns: {np.mean(buy_and_hold_returns)}')

        plt.title(f'Trading evaluation from {start_date} to {end_date}')
        plt.plot(cumsum_return, label='Model Returns')
        plt.plot(buy_and_hold_returns, label='Buy and Hold Returns')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.legend()
        plt.show()

        
