import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(7, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
