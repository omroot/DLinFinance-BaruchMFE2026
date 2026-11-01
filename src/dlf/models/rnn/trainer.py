from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt


class RNNTrainer:
    """
    Trainer class for RNN-based time series models.

    This class handles the training, evaluation, and visualization of RNN models
    for time series prediction tasks.

    Attributes:
        model (nn.Module): The RNN model to be trained
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimization algorithm
        train_dataloader (data.DataLoader): DataLoader for training batches
        train_features (torch.Tensor): Complete training features
        test_features (torch.Tensor): Complete test features
        train_targets (torch.Tensor): Complete training targets
        test_targets (torch.Tensor): Complete test targets
        lookback_window_size (int): Size of the lookback window
        train_size (int): Size of the training dataset
        y (torch.Tensor): Complete time series data
        train_losses (List[float]): History of training losses
        test_losses (List[float]): History of test losses
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            train_dataloader: data.DataLoader,
            train_features: torch.Tensor,
            test_features: torch.Tensor,
            train_targets: torch.Tensor,
            test_targets: torch.Tensor,
            lookback_window_size: int,
            train_size: int,
            y: torch.Tensor
    ) -> None:
        """
        Initialize the RNN trainer.

        Args:
            model: Neural network model to train
            criterion: Loss function for training
            optimizer: Optimization algorithm
            train_dataloader: DataLoader for training batches
            train_features: Complete set of training features
            test_features: Complete set of test features
            train_targets: Complete set of training targets
            test_targets: Complete set of test targets
            lookback_window_size: Size of the lookback window
            train_size: Size of the training dataset
            y: Complete time series data
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.train_features = train_features
        self.test_features = test_features
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.lookback_window_size = lookback_window_size
        self.train_size = train_size
        self.y = y

        self.epoch_losses: List[float] = []
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []

    def train_epoch(self) -> float:
        """
        Run one epoch of training.

        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        self.model.reset_hidden()

        epoch_loss: float = 0.0
        for batch_features, batch_targets in self.train_dataloader:
            self.optimizer.zero_grad()
            y_pred = self.model(batch_features)
            loss = self.criterion(y_pred, batch_targets)
            loss.backward()
            self.optimizer.step()
            self.model.reset_hidden()
            epoch_loss += loss.item()

        return epoch_loss / len(self.train_dataloader)

    def evaluate(self) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Evaluate model on train and test sets.

        Returns:
            Tuple containing:
                - Training predictions
                - Test predictions
                - Training RMSE
                - Test RMSE
        """
        self.model.eval()
        with torch.no_grad():
            self.model.reset_hidden()
            train_y_pred = self.model(self.train_features)
            train_loss = float(self.criterion(train_y_pred, self.train_targets))

            self.model.reset_hidden()
            test_y_pred = self.model(self.test_features)
            test_loss = float(self.criterion(test_y_pred, self.test_targets))

            return train_y_pred, test_y_pred, train_loss, test_loss

    def plot_predictions(
            self,
            train_y_pred: torch.Tensor,
            test_y_pred: torch.Tensor,
            epoch: int
    ) -> None:
        """
        Plot the predictions against actual values.

        Args:
            train_y_pred: Predictions on training data
            test_y_pred: Predictions on test data
            epoch: Current epoch number
        """
        train_plot = np.ones_like(self.y.numpy()) * np.nan
        test_plot = np.ones_like(self.y.numpy()) * np.nan

        train_plot[self.lookback_window_size:self.train_size] = train_y_pred[:, -1, :].squeeze().numpy()
        test_plot[self.train_size + self.lookback_window_size:len(self.y)] = test_y_pred[:, -1, :].squeeze().numpy()

        plt.figure(figsize=(12, 6))
        plt.plot(self.y, label='Actual', alpha=0.8)
        plt.plot(train_plot, c='r', label='Train Predictions', alpha=0.6)
        plt.plot(test_plot, c='g', label='Test Predictions', alpha=0.6)
        plt.title(f'Time Series Predictions - Epoch {epoch}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def train(self, num_epochs: int, plot_every: int = 10) -> None:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            plot_every: Frequency of evaluation and plotting

        Returns:
            List of training losses for each epoch
        """
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()

            train_y_pred, test_y_pred, train_loss, test_loss = self.evaluate()
            self.epoch_losses.append(epoch_loss)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            if epoch % plot_every != 0:
                continue


            print(f"Epoch {epoch}: train RMSE {train_loss:.4f}, test RMSE {test_loss:.4f}")
            self.plot_predictions(train_y_pred, test_y_pred, epoch)
