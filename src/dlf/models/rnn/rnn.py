import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from typing import Union, Type, Tuple, Optional
import matplotlib.pyplot as plt


class RNNType(Enum):
    RNN = "RNN"
    LSTM = "LSTM"
    GRU = "GRU"


class RNN(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 50,
            number_layers: int = 1,
            output_size: int = 1,
            dropout: float = 0.0,
            rnn_type: RNNType = RNNType.RNN
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.rnn_type = rnn_type
        self.batch_first = True

        rnn_cells = {
            RNNType.RNN: nn.RNN,
            RNNType.LSTM: nn.LSTM,
            RNNType.GRU: nn.GRU
        }
        rnn_cell = rnn_cells[rnn_type]

        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=number_layers,
            dropout=dropout if number_layers > 1 else 0,
            batch_first=self.batch_first
        )

        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = None

    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden state."""
        if self.rnn_type == RNNType.LSTM:
            hidden = (
                torch.zeros(self.number_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.number_layers, batch_size, self.hidden_size).to(device)
            )
        else:  # RNN or GRU
            hidden = torch.zeros(self.number_layers, batch_size, self.hidden_size).to(device)
        return hidden

    def reset_hidden(self) -> None:
        """Reset hidden state."""
        self.hidden = None

    def forward(self,
                x: torch.Tensor,
                hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """Forward pass."""
        # batch_size = x.size(0) if self.batch_first else x.size(1)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Add feature dimension if missing

        batch_size = x.size(0)
        # Initialize hidden state if none provided
        if hidden is None:
            self.hidden = self.init_hidden(batch_size, x.device)
        else:
            self.hidden = hidden

        # Run RNN
        rnn_out, self.hidden = self.rnn(x, self.hidden)
        predictions = self.linear(rnn_out)

        return predictions

