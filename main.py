from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    df_train = pd.read_excel('test.xlsx')
    population = GeneticAlgorithm(100, 0, 0, df_train['square'].to_numpy(), df_train['clusters'].to_numpy())
    population.select()