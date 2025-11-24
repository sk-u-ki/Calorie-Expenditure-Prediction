# Prognozowanie spalania kalorii

Prosty model feed-forward w PyTorch na konkurs Kaggle Playground S5E5. Kod trenuje się na `train.csv`, porównuje hiperparametry, generuje wykresy EDA i przygotowuje predykcje w formacie konkursu.

## Model i trening
- Architektura: `Dropout(p=0.4) -> Linear(7,8) -> ReLU -> Linear(8,4) -> ReLU -> Linear(4,1)`, optymalizator SGD + momentum, funkcja straty RMSLE.
- Podział: 80/20 train/val przez `torch.utils.data.random_split`.
- Dropout vs brak dropout: przy `p=0.4` model jest odporniejszy na szum i mniej się przeucza; bez dropout szybciej schodzi na zbiorze treningowym, ale walidacja pogarsza się wcześniej. Wyłączaj tylko przy bardzo czystych/małych danych.
- Najlepszy przebieg z logów poniżej: momentum `0.5`, learning rate `0.001`, 30 epok (RMSLE ~0.10).

## Instalacja środowiska
```bash
git clone <repo_url>
cd Calorie-Expenditure-Prediction
pip install uv              # https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
uv sync
```

## Jak uruchamiać
- Trenowanie jednej konfiguracji: `uv run python -m src.train`
- Siatka hiperparametrów: `uv run python -m src.compare`
- Mapa korelacji: `uv run python -m src.charts.corr` (zapisuje `visual/correlation_matrix.png`)
- Scatter/pairplot: `uv run python -m src.charts.scatter` (zapisuje `visual/scatter_plot.png`)
- Predykcja na teście: `uv run python -m src.predict` (ładuje wagi `outputs/2025-11-22/16-39-52/best_model.pth`)
- Testy jednostkowe: `uv run python -m pytest`

## EDA
- Korelacje: `visual/correlation_matrix.png`
- Scatter/pairplot: `visual/scatter_plot.png`

![Korelacje](visual/correlation_matrix.png)
![Scatter](visual/scatter_plot.png)

## Logi porównania hiperparametrów
```
[2025-11-24 13:20:33,821][__main__][INFO] - TRAINING MODEL with paramentr -> Epoch [30] - MOMENTUM: 0.3 - LEARNING RATE: 0.001
[2025-11-24 13:20:45,884][src.train][INFO] - Epoch [1/30] - RMSLE: 0.85 - TRAIN LOSS: 1.59 - VAL LOSS: 0.85
[2025-11-24 13:22:25,010][src.train][INFO] - Epoch [11/30] - RMSLE: 0.12 - TRAIN LOSS: 0.12 - VAL LOSS: 0.12
[2025-11-24 13:24:13,201][src.train][INFO] - Epoch [21/30] - RMSLE: 0.11 - TRAIN LOSS: 0.11 - VAL LOSS: 0.11
[2025-11-24 13:26:03,372][__main__][INFO] - TRAINING MODEL with paramentr -> Epoch [30] - MOMENTUM: 0.3 - LEARNING RATE: 0.0005
[2025-11-24 13:26:14,568][src.train][INFO] - Epoch [1/30] - RMSLE: 2.24 - TRAIN LOSS: 3.19 - VAL LOSS: 2.23
[2025-11-24 13:28:02,432][src.train][INFO] - Epoch [11/30] - RMSLE: 0.23 - TRAIN LOSS: 0.25 - VAL LOSS: 0.23
[2025-11-24 13:29:53,421][src.train][INFO] - Epoch [21/30] - RMSLE: 0.13 - TRAIN LOSS: 0.13 - VAL LOSS: 0.13
[2025-11-24 13:31:41,874][__main__][INFO] - TRAINING MODEL with paramentr -> Epoch [30] - MOMENTUM: 0.3 - LEARNING RATE: 0.0001
[2025-11-24 13:31:51,624][src.train][INFO] - Epoch [1/30] - RMSLE: 3.41 - TRAIN LOSS: 3.75 - VAL LOSS: 3.41
[2025-11-24 13:33:40,283][src.train][INFO] - Epoch [11/30] - RMSLE: 0.84 - TRAIN LOSS: 0.84 - VAL LOSS: 0.84
[2025-11-24 13:35:29,898][src.train][INFO] - Epoch [21/30] - RMSLE: 0.78 - TRAIN LOSS: 0.78 - VAL LOSS: 0.78
[2025-11-24 13:37:06,410][__main__][INFO] - TRAINING MODEL with paramentr -> Epoch [30] - MOMENTUM: 0.5 - LEARNING RATE: 0.001
[2025-11-24 13:37:16,491][src.train][INFO] - Epoch [1/30] - RMSLE: 0.84 - TRAIN LOSS: 1.70 - VAL LOSS: 0.84
[2025-11-24 13:39:08,396][src.train][INFO] - Epoch [11/30] - RMSLE: 0.11 - TRAIN LOSS: 0.11 - VAL LOSS: 0.11
[2025-11-24 13:41:04,611][src.train][INFO] - Epoch [21/30] - RMSLE: 0.10 - TRAIN LOSS: 0.10 - VAL LOSS: 0.10
[2025-11-24 13:42:42,967][__main__][INFO] - TRAINING MODEL with paramentr -> Epoch [30] - MOMENTUM: 0.5 - LEARNING RATE: 0.0005
[2025-11-24 13:42:54,209][src.train][INFO] - Epoch [1/30] - RMSLE: 1.13 - TRAIN LOSS: 2.49 - VAL LOSS: 1.13
[2025-11-24 13:44:53,941][src.train][INFO] - Epoch [11/30] - RMSLE: 0.19 - TRAIN LOSS: 0.20 - VAL LOSS: 0.19
```

- Konfiguracje treningu są w `src/config`.
- Wykresy zapisują się do `visual/`.
