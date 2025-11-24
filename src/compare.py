import logging
from itertools import product

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from .workout_dataset import WorkoutDataset
from .model import NeuralNetwork
from .train import train

generator = torch.Generator().manual_seed(42)

logger: logging.Logger = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path="config", config_name="config_compare")
def main(
    cfg : DictConfig
) -> None:
    # Przygotowanie DataLoader'a z danymi
    dataset: Dataset = WorkoutDataset(cfg["data"]["train"])

    train_dataset, val_dataset = random_split(
                                            dataset,
                                            [0.8, 0.2],
                                            generator=generator
                                        )
    
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    
    # Pobranie listy hiperparametrów z konfiguracji
    list_epochs: list = cfg["training"]["epochs"]
    list_momentum: list = cfg["training"]["momentum"]
    list_learning_rate: list = cfg["training"]["learning_rate"]

    hiperparam: list = list(product(list_epochs, list_momentum, list_learning_rate))

    # Testowanie różnych kombinacji hiperparametrów
    for epochs, momentum, learning_rate in hiperparam:

        # Zdefiniowanie modelu
        model: nn.Module = NeuralNetwork()

        logger.info(f"TRAINING MODEL with paramentr -> Epoch [{epochs}] - MOMENTUM: {momentum} - LEARNING RATE: {learning_rate}")
    
        # Trenowanie modelu
        train(
            dataset=dataset,
            dataloader=train_loader,
            val_loader=val_loader,
            model=model,
            epochs=epochs,
            lr=learning_rate,
            momentum=momentum
        )
if __name__ == "__main__":
    main()