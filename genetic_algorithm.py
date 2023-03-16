import numpy as np
import matplotlib.pyplot as plt
import time


class GeneticAlgorithm(object):
    def __init__(self,
                 population_size: int,
                 b0: int,
                 b1: int,
                 train: np.ndarray,
                 target: np.ndarray
                 ):
        self.population_size = population_size
        self.target = target
        self.train = train
        self.chromosome = np.ndarray
        self.pred = np.zeros((self.population_size, len(self.train)))
        self.loss = np.zeros(self.population_size)
        self.sigma = 0.1
        self.b0 = np.random.normal(loc=b0, scale=self.sigma, size=self.population_size)
        self.b1 = np.random.normal(loc=b1, scale=self.sigma, size=self.population_size)
        self.individuals = np.transpose(np.array([self.b0, self.b1]))

    def MSE(self, index):
        squares = (self.pred[index, :] - self.target) ** 2
        return squares.mean()

    def select(self):
        count = 0
        plt.ion()
        min_loss = [0, 0]
        changed = True
        while changed:
            count += 1
            print(count)
            changed = False

            self.predict()

            for index in range(self.population_size):
                self.loss[index] = self.MSE(index)

            index_min_loss = np.argmin(self.loss)
            min_loss.pop(0)
            min_loss.append(self.loss[index_min_loss])
            delta = abs(min_loss[0] - min_loss[1])

            print(self.individuals[index_min_loss, :])

            self.chromosome = self.individuals[index_min_loss, :]
            if delta != 0:
                changed = True

            self.b0 = np.random.normal(loc=self.individuals[index_min_loss][0], scale=1, size=self.population_size)
            self.b1 = np.random.normal(loc=self.individuals[index_min_loss][1], scale=1000, size=self.population_size)
            self.individuals = np.transpose(np.array([self.b0, self.b1]))
            self.individuals[0, :] = self.chromosome
            self.print_regression(index_min_loss)

        plt.ioff()
        plt.show()

    def predict(self):
        for i in range(len(self.individuals)):
            for j in range(len(self.train)):
                self.pred[i][j] = self.individuals[i][0] + self.individuals[i][1] * self.train[j]

    def print_regression(self, index):
        plt.clf()
        plt.scatter(self.train, self.target, marker='o', alpha=0.8)
        plt.plot(self.train, self.pred[index, :], 'r')
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.1)