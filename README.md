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
Dropout to metoda regularyzacji, która zmniejsza nadmierne uczenie się. Losowo wyłącza część neuronów podczas uczenia się, zmuszając sieć do niepolegania na poszczególnych połączeniach i tworzenia bardziej stabilnych reprezentacji. Na etapie wnioskowania wszystkie neurony działają, a wagi są korygowane poprzez skalowanie.

Funkcje aktywacji są używane do wprowadzenia nieliniowości do sieci neuronowej, aby mogła ona modelować złożone zależności w danych. Bez nich sieć sprowadzałaby się do jednego przekształcenia liniowego i nie byłaby w stanie rozwiązać większości rzeczywistych zadań. Jeśli w sieci z kilkoma ukrytymi warstwami usunąć funkcje aktywacji, wszystkie jej warstwy zamienią się w jedno duże przekształcenie liniowe. Taka sieć stanie się równoważna jednej warstwie liniowej i straci zdolność modelowania zależności nieliniowych.

Sieci neuronowe są wykorzystywane tam, gdzie zależności są złożone, nieliniowe i trudne do jednoznacznego opisania: w prognozowaniu, klasyfikacji, analizie obrazów, mowy i innych zadaniach z dużą liczbą cech. Nie można zbudować dokładnego predyktora w postaci zestawu if-ów, ponieważ liczba możliwych kombinacji cech wejściowych rośnie wykładniczo i nie da się ręcznie zapisać reguły dla każdej sytuacji. Ponadto rzeczywiste zależności są zazwyczaj ciągłe, z szumem i wariacjami, a zestaw sztywnych warunków nie będzie w stanie uogólniać; gdy pojawi się nowa wartość, nieco różniąca się od znanych, program oparty na if nie będzie wiedział, co zrobić.

Oblicz pochodne parametrów (w i b) dla sieci: wejście 2 cechy [x1, x2], warstwa ukryta 2 neurony + ReLU, wyjście 1 neuron (bez aktywacji), strata MSE, jedna obserwacja [2,3], y=5. Rozważ inicjalizację wag/biasów = 1.0; przy inicjalizacji = 0.0 sieć stoi (symetria, zerowe gradienty).

### Wynik przy w=1.0, b=1.0 (dla powyzszej architektury)
- Dane: `x = [2, 3]`, cel `y = 5`, ReLU w warstwie ukrytej, MSE.
- Forward: `z = W1*x + b1 = [6, 6]`, `h = ReLU(z) = [6, 6]`, `y_hat = W2^T h + b2 = 13`, blad `delta = 8`, strata `L = 32`.
- Gradienty: `dL/dW2 = [48, 48]^T`, `dL/db2 = 8`, `dL/dW1 = [[16, 24], [16, 24]]`, `dL/db1 = [8, 8]^T`, `dL/dz = [8, 8]`.

## Uwagi techniczne
- Konfiguracje treningu sa w `src/config`.
- Wykresy zapisywane sa do `visual/`.
