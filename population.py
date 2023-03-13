import numpy as np


class Population(object):
    def __init__(self, population_size: int, b0: int, b1: int):
        self.population_size = population_size
        self.sigma = 0.1
        self.b0 = np.random.normal(loc=b0, scale=self.sigma, size=self.population_size)
        self.b1 = np.random.normal(loc=b1, scale=self.sigma, size=self.population_size)
        self.individuals = np.transpose(np.array([self.b0, self.b1]))


if __name__ == "__main__":
    pop = Population(10, 0, 0)
    print(pop.individuals)