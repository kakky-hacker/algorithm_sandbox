import random
from typing import List

from individual import Individual


def random_int(a: int, b: int, p1: float, p2: float):
    def random_int_core(population: List[Individual]):
        for individual in population:
            if p1 <= random.random():
                continue
            for i in range(len(individual.chromosome)):
                if p2 <= random.random():
                    continue
                individual.chromosome[i] = random.randint(a, b)

    return random_int_core
