from collections import OrderedDict
from dataclasses import dataclass
import re
import pandas as pd
from scipy.stats import norm
from torch import device, nn, Tensor
from typing import Dict, List

import torch

from src.models.utils import get_indicator, sliding_window

# run on CUDA iff it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass(frozen=True)
class CustomFeatureData:
    """
    A class for keeping track of feature-related data.
    """
    feature_name: str
    input_size: int
    hidden_size: int
    batch_first: bool = True


@dataclass(frozen=True)
class CustomFeature:
    """
    A class for keeping track of custom features.
    """
    feature_data: CustomFeatureData
    data: Tensor


class FeatureMissingException(Exception):
    """
    A custom exception for missing features.
    """

    def __init__(self, message="That feature is missing!"):
        super().__init__(message)


class Custom(nn.Module):
    """
    Our custom model for performing buy/sell predictions.
    """

    def __init__(self,
                 features: List[CustomFeatureData],
                 trading_indicator_input_size: int,
                 rnn_aggregation_hidden_size: int,
                 trading_indicator_hidden_size: int,
                 linear_aggregator_hidden_size: int):
        """
        Constructs a new instance of the Custom class.

        Args:

        :param features: The input features (i.e. closing, and other things like that).
        :param rnn_aggregation_hidden_size: The hidden size of the rnn aggregation linear layer.
        :param trading_indicator_input_size: The number of trading indicators to take into account.
        :param trading_indicator_hidden_size: The hidden size for the linear layer that takes these trading
                                              indicators as input.
        :param linear_aggregator_hidden_size: The hidden size of the aggregator layer that aggregates both the linear layer
                                              for the indicators and linear layer that took the RNN outputs as its input.
        """
        super(Custom, self).__init__()

        # create a list of LSTMs
        rnn_dict: Dict[str, nn.RNN] = {}
        for feature in features:
            rnn = nn.LSTM(
                input_size=feature.input_size,
                hidden_size=feature.hidden_size,
                num_layers=1
            )
            rnn.name = feature.feature_name
            rnn_dict[feature.name] = rnn

        # define the model's layers
        self.rnn_dict = rnn_dict
        self.rnn_aggregation = nn.Linear(
            sum(map(lambda x: x.hidden_size * x.num_layers, self.rnn_dict.values())), rnn_aggregation_hidden_size)
        self.linear_indicator = nn.Linear(
            trading_indicator_input_size, trading_indicator_hidden_size)
        self.linear_aggregator = nn.Linear(
            trading_indicator_hidden_size + rnn_aggregation_hidden_size, linear_aggregator_hidden_size)
        self.final_linear_layer = nn.Linear(linear_aggregator_hidden_size, 1)

    def forward(self, features: List[CustomFeature], indicators: Tensor) -> Tensor:
        """
        The forward method, defines how the model
        shall be run.

        Args:

        :param features: The input features (closing, etc...).
        :param indicators: Trade indicators.
        :param value_at_risk: The value at risk. 
        :return: An output tensor which results after the input data 
                 goes through the model. This is our prediction for
                 the number of shares to buy.
        """
        rnn_outputs = []
        for feature in features:
            feature_data = feature.feature_data
            if feature_data.feature_name not in self.rnn_dict.keys():
                raise FeatureMissingException(
                    f'The feature [<{feature_data.feature_name}>] does not exist!')

            # h has dim = (num_layers, h_out)
            _, h, _ = self.rnn_dict[feature_data.feature_name](feature.data)
            rnn_outputs.append(h)

        concatenated_outputs = torch.cat(tuple(map(lambda h: h[0], rnn_outputs)), dim=1)
        linear_from_rnn = torch.relu(self.rnn_aggregation(concatenated_outputs))
        linear_indicators = torch.relu(self.linear_indicator(indicators))
        indicators_and_linear_output = torch.cat(linear_from_rnn, linear_indicators, dim=1)
        linear_aggregator = torch.relu(self.linear_aggregator(indicators_and_linear_output))
        return torch.sigmoid(self.final_linear_layer(linear_aggregator)).view(-1)


def train_model(data: pd.DataFrame,
                start_date: str,
                end_date: str,
                parameters: Dict,
                window_size: int = 40,
                num_epochs: int = 500,
                train_val_ratio: float = 0.8):
    """TODO"""
    # cut after end date
    data = data[data.index < end_date]
    ts_len = data[data.index > start_date].shape[0]
    train_length = int(ts_len * train_val_ratio)

    # get the hyperparameters
    learning_rate = parameters['lr']
    rnn_aggregation_hidden_size = parameters['rnn_agg_hidden_size']
    trading_indicator_hidden_size = parameters['trading_ind_hidden_size']
    linear_aggregator_hidden_size = parameters['linear_agg_hidden_size']
    ind1_name = parameters['ind1']['_name']
    ind2_name = parameters['ind2']['_name']

    # an ordered dictionary of data
    data_source = OrderedDict()
    for i in data.columns:
        key_prefix = i.strip().replace(" ", "_").lower()
        data_source[f'{key_prefix}']
        data_source[f'{key_prefix}']
        i: str = i
        print(i.strip().replace(" ", "_").lower())
    
    data_source['close_diff'] = (data['Close'] - data['Close'].shift(1))
    data_source['close_roc'] = (data['Close'] / data['Close'].shift(1))
    data_source['ind1'] = get_indicator(data, ind1_name, parameters['ind1'])
    data_source['ind2'] = get_indicator(data, ind2_name, parameters['ind2'])

    # add value at risk as an indicator
    pct_changes = data['Close'].pct_change()
    value_at_risk = []
    for i in range(len(pct_changes)):
        mean = pct_changes[:i].mean()
        std = pct_changes[:i].std()
        value_at_risk_pct = abs(norm.ppf(0.01, mean, std))
        value_at_risk.append(value_at_risk_pct)

    data_source['var'] = pd.Series(value_at_risk)
    data_source['var'].index = pct_changes.index

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




    


def evaluate_model():
    """TODO"""
    pass
