from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":
    df_train = pd.read_excel('test.xlsx')
    X = df_train['square'].to_numpy()
    Y = df_train['clusters'].to_numpy()

    population = GeneticAlgorithm(
        population_size=100,
        b0=0,
        b1=0,
        sigma_b0=0.1,
        sigma_b1=1000,
        train=X,
        target=Y)
    population.select()

    x_loss = np.arange(len(population.loss_history))
    print(len(x_loss))
    print(len(population.loss_history))
    plt.plot(x_loss, population.loss_history, label='loss')
    plt.legend()
    plt.show()