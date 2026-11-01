import torch
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod


class TimeSeriesPreprocessor(ABC):
    """Abstract base class for time series preprocessors."""

    def __init__(self, lookback_window_size: int):
        if lookback_window_size < 1:
            raise ValueError("Window size must be positive")
        self.lookback_window_size = lookback_window_size

    @abstractmethod
    def create_prediction_windows(self, time_series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sliding windows from time series data."""
        pass