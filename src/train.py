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


logger: logging.Logger = logging.getLogger(__name__)


def train(
    dataset: Dataset,
    dataloader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 100,
    lr: float = 1e-3,
    momentum: float = 0.9
) -> None:
    # Funkcja kosztu
    criterion: nn.Module = RMSLELoss()

    # Algorytm optymalizacyjny
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_rmsle: float = float("inf")
    losses: list[float] = []
    # Trening przez zdefiniowaną liczbę epok
    for epoch in range(epochs):
        running_loss: float = 0.0
        # Iteracja po wszystkich batchach w zbiorze treninigowym
        for X_batch, y_batch in dataloader:
            # Wyzeruj wszystkie gradienty, w przeciwnym razie stare gradienty będą się akumulować
            optimizer.zero_grad()

            # Predykcja modelu
            logits = model(X_batch)

            # Obliczenie funkcji straty
            loss = criterion(logits, y_batch)
            running_loss += loss.item()

            # Obliczenie gradientów
            loss.backward()

            # Aktualizacja wag na podstawie obliczonych gradientów
            optimizer.step()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                pred_val = model(x_val)
                loss_val = criterion(pred_val, y_val)
                val_loss += loss_val.item()

        val_loss /= len(val_loader)
        
        running_loss /= len(dataloader)
        losses.append(running_loss)

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred: torch.Tensor = model(dataset.x)
                rmsle: torch.Tensor = criterion(pred, dataset.y)
                logger.info(f"Epoch [{epoch + 1}/{epochs}] - RMSLE: {rmsle:.2f} - TRAIN LOSS: {running_loss:.2f} - VAL LOSS: {val_loss:.2f}")
                if rmsle.item() < best_rmsle:
                    best_rmsle = rmsle.item()
                    save_path: str = os.path.join(HydraConfig.get().run.dir, "best_model.pth")
                    torch.save(model.state_dict(), save_path)

    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.show(block=False)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    # Przygotowanie DataLoader'a z danymi
    dataset: Dataset = WorkoutDataset(cfg["data"]["train"])

    train_dataset, val_dataset = random_split(
                                            dataset,
                                            [0.8, 0.2],
                                            generator=generator
                                        )
    
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)


    # Zdefiniowanie modelu
    model: nn.Module = NeuralNetwork()

    # Trenowanie modelu
    train(
        dataset=dataset,
        dataloader=train_loader,
        val_loader=val_loader,
        model=model,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"]
    )


if __name__ == "__main__":
    main()
