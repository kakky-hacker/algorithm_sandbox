from typing import List

from individual import Individual


class GeneticAlgorithm:
    def __init__(
        self,
        direction: str,
        objective_func,
        generate_func,
        selction_func,
        crossover_func,
        mutation_func,
    ):
        self.direction = direction
        self.objective_func = objective_func
        self.generate_func = generate_func
        self.selction_func = selction_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func

    def solve(self, n_generations: int, n_population: int):
        population = [
            Individual.instantiate(
                self.generate_func, self.objective_func, self.direction
            )
            for _ in range(n_population)
        ]
        for _ in range(n_generations):
            population = self.selction_func(population)
            children = self.crossover_func(
                population, self.objective_func, n_population - len(population)
            )
            children = self.mutation_func(children)
            population.extend(children)
        return self.selction_func(population)
