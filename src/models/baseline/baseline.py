"""
Author: Nathan Waltz

Implemented a baseline for the model.
"""

from darts.models.forecasting.baselines import NaiveDrift
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts import TimeSeries


class Baseline:
    """
    A wrapper class for the ARIMA baseline.
    """

    def __init__(self):
        """
        Constructs a new instance of the Baseline class.
        """
        self.baseline_name = 'Naive Drift'
        self.model = NaiveDrift()

    def fit(self, dataset: TimeSeries) -> ForecastingModel:
        """
        Creates a new model from a time series dataset.

        :param dataset: The time series dataset to use.
        :return: The fitted model.
        """
        return self.model.fit(series=dataset)

    def get_baseline_name(self) -> str:
        """
        :return: The name of the baseline model.
        """
        return self.baseline_name

    @staticmethod
    def predict(forecast_horizon: int, model: ForecastingModel) -> TimeSeries:
        """
        Performs a prediction using the AutoARIMA model over
        a given forecast horizon.

        :param forecast_horizon: The forecast horizon to forecast for.
        :param model: The model to predict with.
        :return: A time series containing the next n points after the
                end of the training series.
        """
        return model.predict(forecast_horizon)
