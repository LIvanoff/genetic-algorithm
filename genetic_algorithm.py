import numpy as np
import matplotlib.pyplot as plt
import time


class GeneticAlgorithm(object):
    loss_history: np.ndarray

    def __init__(self,
                 population_size: int,
                 b0: int,
                 b1: int,
                 sigma_b0: float,
                 sigma_b1: float,
                 train: np.ndarray,
                 target: np.ndarray
                 ):
        self.population_size = population_size
        self.target = target
        self.train = train
        self.chromosome = np.ndarray
        self.pred = np.zeros((self.population_size, len(self.train)))
        self.loss = np.zeros(self.population_size)
        self.sigma_b0 = sigma_b0
        self.sigma_b1 = sigma_b1
        self.b0 = np.random.normal(loc=b0, scale=self.sigma_b0, size=self.population_size)
        self.b1 = np.random.normal(loc=b1, scale=self.sigma_b1, size=self.population_size)
        self.individuals = np.transpose(np.array([self.b0, self.b1]))

    def MSE(self, index):
        squares = (self.pred[index, :] - self.target) ** 2
        return squares.mean()

    def select(self):
        count = 0
        plt.ion()
        min_loss = [0, 0]
        self.loss_history = np.array([])
        changed = True
        while changed:
            count += 1
            print(count)
            changed = False

            self.predict()

            for index in range(self.population_size):
                self.loss[index] = self.MSE(index)

            index_min_loss0 = np.argmin(self.loss)
            val0, val1 = np.partition(self.loss, 1)[0:2]
            index = np.where(self.loss == val1)
            index_min_loss1 = index[0][0]
            min_loss.pop(0)
            min_loss.append(self.loss[index_min_loss0])
            self.loss_history = np.append(self.loss_history, min_loss[1])
            delta = abs(min_loss[0] - min_loss[1])

            self.chromosome = self.individuals[index_min_loss0, :]
            if delta != 0:
                changed = True

            # crossover
            fun_value0 = self.individuals[index_min_loss0][0] + np.multiply(self.train,
                                                                            self.individuals[index_min_loss1][1])
            fun_value1 = self.individuals[index_min_loss1][0] + np.multiply(self.train,
                                                                            self.individuals[index_min_loss0][1])

            squares0 = np.mean(np.power((fun_value0 - self.target), 2))
            squares1 = np.mean(np.power((fun_value1 - self.target), 2))

            index_min_loss = np.array([])

            if squares1 >= squares0:
                index_min_loss = np.append(index_min_loss, index_min_loss0)
                index_min_loss = np.append(index_min_loss, index_min_loss1)
            else:
                index_min_loss = np.append(index_min_loss, index_min_loss1)
                index_min_loss = np.append(index_min_loss, index_min_loss0)

            self.b0 = np.random.normal(loc=self.individuals[int(index_min_loss[0])][0], scale=self.sigma_b0,
                                       size=self.population_size)
            self.b1 = np.random.normal(loc=self.individuals[int(index_min_loss[1])][1], scale=self.sigma_b1,
                                       size=self.population_size)
            self.individuals = np.transpose(np.array([self.b0, self.b1]))
            self.individuals[0, :] = self.chromosome
            self.print_regression(int(index_min_loss[0]))
            print(min_loss)

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
        plt.title('y = ' + str(self.b1[index]) + ' x + ' + str(self.b0[index]) + ' + ' + str(self.loss[-1]),
                  fontsize=10, color='0.5')
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.01)
