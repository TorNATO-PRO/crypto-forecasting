import torch
import torch.nn as nn

from scipy.stats import norm

"""
The loss functions that will be used throughout our application.
"""

class NegativeMeanReturnLoss(nn.Module):
    """
    Negative mean return loss, negative to maximize the
    mean return on investments.
    """

    def __init__(self):
        """
        Constructs a new instance of the NegativeMeanReturnLoss class.
        """
        super(NegativeMeanReturnLoss, self).__init__()

    def forward(self, lots: torch.Tensor, price_diff: torch.Tensor) -> torch.Tensor:
        """
        The forward method, defines how the model
        shall be run.

        :param lots: The model's proposed stock holdings.
        :param price_diff: The difference in price from the day before, computed
        for each day.
        :return: The negative mean return loss.
        """
        abs_return = torch.mul(lots.view(-1), price_diff)
        ar = torch.mean(abs_return)
        return torch.neg(ar)


class RiskAdjustedMeanLoss(nn.Module):
    """
    Value at risk adjusted mean loss.
    """

    def __init__(self):
        """
        Constructs a new instance of the RiskAdjustedMeanLoss class.
        """
        super(RiskAdjustedMeanLoss, self).__init__()

    def forward(self, lots: torch.Tensor, price_diff: torch.Tensor) -> torch.Tensor:
        """
        The forward method, defines how the model
        shall be run.

        :param lots: The model's proposed stock holdings.
        :param price_diff: The difference in price from the day before, computed
        for each day.
        :return: The negative mean return loss.
        """
        return torch.neg(torch.mean(torch.mul(lots.view(-1), price_diff)))
