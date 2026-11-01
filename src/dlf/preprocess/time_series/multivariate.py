import torch
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod

from dlf.preprocess.time_series.base import TimeSeriesPreprocessor


class MultivariatePreprocessor(TimeSeriesPreprocessor):
    """Preprocessor for multivariate time series prediction."""

    def __init__(self, window_size: int, target_dim: int):
        """
        Args:
            window_size: Number of time steps to use as input
            target_dim: Index of the target dimension to predict
        """
        super().__init__(window_size)
        self.target_dim = target_dim

    def create_prediction_windows(
            self,
            time_series: torch.Tensor  # shape: (series_length, num_features)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert multivariate time series into sliding windows for prediction.

        Args:
            time_series: 2D tensor of shape (series_length, num_features)

        Returns:
            Tuple containing:
                - features: tensor of shape (num_windows, window_size, num_features)
                - targets: tensor of shape (num_windows, 1)
        """
        if len(time_series.shape) != 2:
            raise ValueError("Expected 2D tensor for multivariate time series")

        series_length, num_features = time_series.shape
        if self.target_dim >= num_features:
            raise ValueError(f"Target dimension {self.target_dim} exceeds number of features {num_features}")

        if series_length <= self.window_size:
            raise ValueError(
                f"Time series length ({series_length}) must be greater than window size ({self.window_size})")

        windows = []
        targets = []

        max_start_idx = series_length - self.window_size
        for start_idx in range(max_start_idx):
            end_idx = start_idx + self.window_size
            target_idx = end_idx

            windows.append(time_series[start_idx:end_idx])
            targets.append(time_series[target_idx:target_idx + 1, self.target_dim])

        return torch.stack(windows), torch.stack(targets)

