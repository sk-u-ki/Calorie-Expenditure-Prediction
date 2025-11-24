import os
import logging
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .workout_dataset import WorkoutDataset, TestDataset
from .model import NeuralNetwork
from .rmsle_loss import RMSLELoss

generator = torch.Generator().manual_seed(42)

def save_predictions(predictions: torch.Tensor, id: pd.DataFrame,  output_file: str) -> None:
    with open(output_file, "w") as f:
        f.write("ID,Calories\n")  # заголовок
        for v, i in zip(predictions, id):
            f.write(f"{i},{v.item()}\n")



def predict(
    dataset: Dataset,
    model: nn.Module,
) -> torch.Tensor:
    
    model.eval()
    
    with torch.no_grad():
        pred: torch.Tensor = model(dataset.x)
    return pred


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    # Przygotowanie DataLoader'a z danymi
    dataset: Dataset = TestDataset(cfg["data"]["test"])

    # Zdefiniowanie modelu
    model: nn.Module = NeuralNetwork()

    # Załadowanie wytrenowanego modelu
    # Trzeba pamiętać o architekturze modelu
    model.load_state_dict(torch.load(cfg["data"]["best_model"]))

    # Ustawienie modelu w tryb ewaluacji
    

    # Trenowanie modelu
    pred: torch.Tensor = predict(
                                    dataset=dataset,
                                    model=model
                                )
    
    save_predictions(predictions=pred, id = dataset.id, output_file="data/submission.csv")
    


if __name__ == "__main__":
    main()
