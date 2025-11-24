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

## Szybki start
- `git checkout -b <twoj_branch>` po klonie.
- `uv sync`
- Trening: `uv run python -m src.train`
- Grid: `uv run python -m src.compare`
- Testy: `uv run python -m pytest`
- Seed: ustawiony na 42 (generator w train/compare) dla powtarzalnosci.

## Wymagania projektu (checklista)
- Podzial train/val (80/20) z ustawionym seedem.
- Porownanie hiperparametrow (lr, momentum, epoki) oraz dropout vs bez dropout.
- Skrypt predykcji w formacie konkursu.
- Wizualizacje danych: korelacje i scatter/pairplot zapisane w `visual/`.

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

## Zadanie teoretyczne — obliczenia
![Gradients 1](visual/0f76e0aa-c620-4fd9-8179-5b10b4a6246c.jpg)
![Gradients 2](visual/4351468d-b808-489b-b20b-7ce1d74f8c60.jpg)
![Gradients 3](visual/a31ba1c1-9356-4931-a8f8-9624cac1bb6a.jpg)

## Zadanie teoretyczne
- Oblicz pochodne parametrow (w i b) sieci: wejscie 2 cechy [x1, x2], warstwa ukryta 2 neurony + ReLU, wyjscie 1 neuron (bez aktywacji), strata MSE, jedna obserwacja [2,3], y=5. Sprawdz przypadek inicjalizacji wszystkich wag/biasow = 1.0; przy inicjalizacji = 0.0 siec stoi (symetria, zerowe gradienty).
- Pytania:  
  • Dla jakich zadan warto stosowac sieci i dlaczego nie da sie napisac recznie programu z if-ami dla wszystkich kombinacji.  
  • Po co funkcje aktywacji i co sie stanie, gdy w wielowarstwowej sieci je usunac (redukcja do pojedynczej transformacji liniowej).  
  • Rola dropout: metoda regularyzacji, losowe wylaczanie neuronow w treningu, mniejsze przeuczenie.

### Wynik przy w=1.0, b=1.0 (dla powyzszej architektury)
- Dane: `x = [2, 3]`, cel `y = 5`, ReLU w warstwie ukrytej, MSE.
- Forward: `z = W1*x + b1 = [6, 6]`, `h = ReLU(z) = [6, 6]`, `y_hat = W2^T h + b2 = 13`, blad `delta = 8`, strata `L = 32`.
- Gradienty: `dL/dW2 = [48, 48]^T`, `dL/db2 = 8`, `dL/dW1 = [[16, 24], [16, 24]]`, `dL/db1 = [8, 8]^T`, `dL/dz = [8, 8]`.

## Uwagi techniczne
- Konfiguracje treningu sa w `src/config`.
- Wykresy zapisywane sa do `visual/`.
