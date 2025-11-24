from matplotlib import pyplot as plt
import seaborn as sns



import pandas as pd

def scatter_plot(data, block=False):
    sns.pairplot(data, height=1.2)
    plt.savefig("visual/scatter_plot.png", dpi=300, bbox_inches='tight')
    

def main() -> None:
    data = pd.read_csv("data/train.csv")
    data = data.drop(columns=["id", 'Sex'], axis=1)
    scatter_plot(data, block=True)

if __name__ == "__main__":
    main()