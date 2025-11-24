from matplotlib import pyplot as plt
import pandas as pd

def corr_plot(data):
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title('Correlation Matrix')
    plt.savefig("visual/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

def main() -> None:
    data = pd.read_csv("data/train.csv")
    data = data.drop(columns=["id", 'Sex'], axis=1)
    corr_plot(data)

if __name__ == "__main__":
    main()