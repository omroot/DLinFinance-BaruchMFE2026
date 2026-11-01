import torch
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod


from dlf.preprocess.time_series.base import TimeSeriesPreprocessor

class UnivariatePreprocessor(TimeSeriesPreprocessor):
    """Preprocessor for univariate time series prediction."""

    def create_prediction_windows(
            self,
            time_series: torch.Tensor  # shape: (series_length,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert univariate time series into sliding windows for prediction.

        Args:
            time_series: 1D tensor of shape (series_length,)

        Returns:
            Tuple containing:
                - features: tensor of shape (num_windows, window_size)
                - targets: tensor of shape (num_windows, 1)
        """
        if len(time_series.shape) != 1:
            raise ValueError("Expected 1D tensor for univariate time series")

        series_length = len(time_series)
        if series_length <= self.lookback_window_size:
            raise ValueError(
                f"Time series length ({series_length}) must be greater than window size ({self.lookback_window_size})")
        if len(time_series.shape) == 1:
            time_series = time_series.unsqueeze(-1)
        features = []
        targets = []

        max_start_idx = series_length - self.lookback_window_size
        for start_idx in range(max_start_idx):
            end_idx = start_idx + self.lookback_window_size
            target_idx = end_idx
            x = time_series[start_idx:end_idx]
            y = time_series[target_idx:target_idx + 1]
            features.append(x)
            targets.append(y)

        return torch.stack(features), torch.stack(targets)

