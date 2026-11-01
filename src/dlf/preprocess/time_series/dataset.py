import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            features: Tensor of shape (num_windows, window_size)
            targets: Tensor of shape (num_windows, 1)
        """
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]