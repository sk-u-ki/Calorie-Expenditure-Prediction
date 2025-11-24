import os
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .workout_dataset import WorkoutDataset
from .model import NeuralNetwork
from .rmsle_loss import RMSLELoss


logger: logging.Logger = logging.getLogger(__name__)


def test(
    dataset: Dataset,
    model: nn.Module,
) -> None:
    criterion: nn.Module = RMSLELoss()
    with torch.no_grad():
        pred: torch.Tensor = model(dataset.x)
        rmsle: torch.Tensor = criterion(pred, dataset.y)
        logger.info(f"TEST RMSLE: {rmsle:.2f}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    # Przygotowanie DataLoader'a z danymi
    dataset: Dataset = WorkoutDataset(cfg["data"]["test"])

    # Zdefiniowanie modelu
    model: nn.Module = NeuralNetwork()

    # Załadowanie wytrenowanego modelu
    model.load_state_dict(torch.load("outputs/2025-11-22/16-39-52/best_model.pth"))

    # Ustawienie modelu w tryb ewaluacji
    model.eval()

    # Trenowanie modelu
    test(
        dataset=dataset,
        model=model
    )


if __name__ == "__main__":
    main()
