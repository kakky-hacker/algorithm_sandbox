import random
from copy import deepcopy
from typing import List

from individual import Individual


def one_point():
    def one_point_func_core(population: List[Individual], objective_func, n_children):
        children = []
        for _ in range(max(n_children // 2, 1)):
            parent1, parent2 = random.sample(population, 2)
            idx = random.randint(0, len(parent1.chromosome) - 1)
            child_chromosome1 = deepcopy(
                parent1.chromosome[:idx] + parent2.chromosome[idx:]
            )
            child_chromosome2 = deepcopy(
                parent2.chromosome[:idx] + parent1.chromosome[idx:]
            )
            children.append(
                Individual(
                    child_chromosome1,
                    objective_func(child_chromosome1),
                    parent1.direction,
                )
            )
            children.append(
                Individual(
                    child_chromosome2,
                    objective_func(child_chromosome1),
                    parent1.direction,
                )
            )
        return children

    return one_point_func_core


def two_points():
    def two_points_func_core(population: List[Individual], objective_func, n_children):
        children = []
        for _ in range(max(n_children // 2, 1)):
            parent1, parent2 = random.sample(population, 2)
            idx1 = random.randint(0, len(parent1.chromosome) - 1)
            idx2 = random.randint(0, len(parent1.chromosome) - 1)
            if idx2 < idx1:
                idx1, idx2 = idx2, idx1
            child_chromosome1 = deepcopy(
                parent1.chromosome[:idx1]
                + parent2.chromosome[idx1:idx2]
                + parent1[idx2:]
            )
            child_chromosome2 = deepcopy(
                parent2.chromosome[:idx1]
                + parent1.chromosome[idx1:idx2]
                + parent2.chromosome[idx2:]
            )
            children.append(
                Individual(
                    child_chromosome1,
                    objective_func(child_chromosome1),
                    parent1.direction,
                )
            )
            children.append(
                Individual(
                    child_chromosome2,
                    objective_func(child_chromosome1),
                    parent1.direction,
                )
            )
        return children

    return two_points_func_core


def uniform_crossover(mask: List[bool]):
    def uniform_crossover_core(
        population: List[Individual], objective_func, n_children
    ):
        children = []
        for _ in range(max(n_children // 2, 1)):
            parent1, parent2 = random.sample(population, 2)
            child_chromosome1 = deepcopy(
                [
                    parent2.chromosome[i] if mask[i] else parent1.chromosome[i]
                    for i in range(len(parent1.chromosome))
                ]
            )
            child_chromosome2 = deepcopy(
                [
                    parent1.chromosome[i] if mask[i] else parent2.chromosome[i]
                    for i in range(len(parent1.chromosome))
                ]
            )
            children.append(
                Individual(
                    child_chromosome1,
                    objective_func(child_chromosome1),
                    parent1.direction,
                )
            )
            children.append(
                Individual(
                    child_chromosome2,
                    objective_func(child_chromosome1),
                    parent1.direction,
                )
            )
        return children

    return uniform_crossover_core
