import os
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .workout_dataset import WorkoutDataset
from .model import NeuralNetwork
from .rmsle_loss import RMSLELoss

generator = torch.Generator().manual_seed(42)


def predict(
    dataset: Dataset,
    model: nn.Module,
) -> torch.Tensor:
    
    model.eval()
    
    model.pr
    with torch.no_grad():
        pred: torch.Tensor = model(dataset.x)
    return pred


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    # Przygotowanie DataLoader'a z danymi
    dataset: Dataset = WorkoutDataset(cfg["data"]["test"])

    # Zdefiniowanie modelu
    model: nn.Module = NeuralNetwork()

    # Załadowanie wytrenowanego modelu
    model.load_state_dict(torch.load("outputs/2025-11-22/16-39-52/best_model.pth"))

    # Ustawienie modelu w tryb ewaluacji
    

    # Trenowanie modelu
    test(
        dataset=dataset,
        model=model
    )


if __name__ == "__main__":
    main()
