from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    pop = GeneticAlgorithm(10, 0, 0)
    df_train = pd.read_excel('test.xlsx')

    plt.scatter(df_train['square'], df_train['clusters'], marker='o', alpha=0.8)
    plt.show()