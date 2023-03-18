from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    df_train = pd.read_excel('test.xlsx')
    population = GeneticAlgorithm(
        population_size=100,
        b0=0,
        b1=0,
        sigma_b0=0.1,
        sigma_b1=1000,
        train=df_train['square'].to_numpy(),
        target=df_train['clusters'].to_numpy())
    population.select()