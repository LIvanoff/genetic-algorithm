from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    df_train = pd.read_excel('test.xlsx')
    population = GeneticAlgorithm(10, 0, 0, df_train['square'].to_numpy(), df_train['clusters'].to_numpy())

    # plt.scatter(df_train['square'], df_train['clusters'], marker='o', alpha=0.8)
    # plt.plot(pop.individuals[:, 0], pop.individuals[:, 1])
    population.select()
    # plt.show()