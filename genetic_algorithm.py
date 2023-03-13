from population import Population


class GeneticAlgorithm(Population):
    def __init__(self, population_size: int, b0: int, b1: int):
        super().__init__(population_size, b0, b1)

