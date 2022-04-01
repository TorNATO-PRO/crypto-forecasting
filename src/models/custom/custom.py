from dataclasses import dataclass
from torch import nn, Tensor
from typing import List

import torch


@dataclass(frozen=True)
class CustomFeatureData:
    """
    A class for keeping track of feature-related data.
    """
    feature_name: str
    input_size: int
    hidden_size: int
    num_layers: int
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
        self.message = message
        super().__init__(self.message)


class Custom(nn.Module):
    """
    Our custom model for performing buy/sell predictions.
    """

    def __init__(self,
                 features: List[CustomFeatureData],
                 rnn_aggregation_hidden_size: int,
                 trading_indicator_input_size: int,
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
        rnn_dict = {}
        for feature in features:
            rnn = nn.LSTM(
                input_size=feature.input_size,
                hidden_size=feature.hidden_size,
                num_layers=feature.num_layers
            )
            rnn.name = feature.feature_name
            rnn_dict[feature.name] = rnn

        # define the model's layers
        self.rnn_dict = rnn_dict
        self.rnn_aggregation = nn.Linear(
            sum(map(lambda x: x.hidden_size, self.rnn_dict.values())), rnn_aggregation_hidden_size)
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
                    f'The feature [{feature_data.feature_name}] does not exist!')

            _, h, _ = self.rnn_dict[feature_data.feature_name](feature.data)
            rnn_outputs.append(h)

        concatenated_outputs = torch.cat(tuple(map(lambda h: h[0], rnn_outputs)), dim=1)
        linear_from_rnn = torch.relu(self.rnn_aggregation(concatenated_outputs))
        linear_indicators = torch.relu(self.linear_indicator(indicators))
        indicators_and_linear_output = torch.cat(linear_from_rnn, linear_indicators, dim=1)
        linear_aggregator = torch.relu(self.linear_aggregator(indicators_and_linear_output))
        return torch.sigmoid(self.final_linear_layer(linear_aggregator)).view(-1)
