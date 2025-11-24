# Prognozowanie spalania kalorii

Prosty model feed-forward w PyTorch na konkurs Kaggle Playground S5E5. Kod trenuje na `train.csv`, porownuje hiperparametry, generuje wykresy EDA i przygotowuje predykcje w formacie konkursu.

## Model i trening
- Architektura: `Dropout(p=0.4) -> Linear(7,8) -> ReLU -> Linear(8,4) -> ReLU -> Linear(4,1)`, optymalizator SGD + momentum, funkcja straty RMSLE.
- Podzial: 80/20 train/val przez `torch.utils.data.random_split`.
- Dropout vs brak dropout: przy `p=0.4` model jest odporniejszy na szum i mniej sie przeucza; bez dropout szybciej schodzi na zbiorze treningowym, ale walidacja pogarsza sie wczesniej. Wylaczaj tylko przy bardzo czystych/malych danych.
- Wnioski z porownania (30 epok): najlepsze lr `0.001`, momentum `0.5` (RMSLE ~0.10, train/val spojne). lr `0.001` i momentum `0.3` bardzo blisko (~0.11). lr `0.0005` za wolne na 30 epok (>=0.13), lr `0.0001` nie schodzi (~0.78). Brak oznak overfittingu; dropout 0.4 stabilizuje trening.

## Instalacja srodowiska
```bash
git clone <repo_url>
cd Calorie-Expenditure-Prediction
pip install uv              # https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
uv sync
```

## Jak uruchamiac
- Trening jednej konfiguracji: `uv run python -m src.train`
- Siatka hiperparametrow: `uv run python -m src.compare`
- Mapa korelacji: `uv run python -m src.charts.corr` (zapisuje `visual/correlation_matrix.png`)
- Scatter/pairplot: `uv run python -m src.charts.scatter` (zapisuje `visual/scatter_plot.png`)
- Predykcja na tescie: `uv run python -m src.predict` (laduje wagi `outputs/2025-11-22/16-39-52/best_model.pth`)
- Testy jednostkowe: `uv run python -m pytest`

## EDA
- Korelacje: `visual/correlation_matrix.png`
- Scatter/pairplot: `visual/scatter_plot.png`

![Korelacje](visual/correlation_matrix.png)
![Scatter](visual/scatter_plot.png)

## Uwagi techniczne
- Konfiguracje treningu sa w `src/config`.
- W `src.predict` jest literowka (`model.pr`) i wywolanie nieistniejacej funkcji `test`; przed uzyciem popraw to albo uruchom `uv run python -m src.test`, by sprawdzic zapisany model.
- Wykresy zapisywane sa do `visual/`.
