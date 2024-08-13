import random
from typing import List

from individual import Individual


def one_point():
    def one_point_func_core(population: List[Individual], objective_func, n_children):
        children = []
        for _ in range(n_children):
            parent1, parent2 = random.sample(population, 2)
            idx = random.randint(0, len(parent1.chromosome) - 1)
            child_chromosome = parent1.chromosome[:idx] + parent2.chromosome[idx:]
            children.append(
                Individual(
                    child_chromosome,
                    objective_func(child_chromosome),
                    parent1.direction,
                )
            )
        return children

    return one_point_func_core


def two_points():
    pass
