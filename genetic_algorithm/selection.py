from typing import List

from individual import Individual


def roulette():
    pass


def elite(num):
    def elite_func_core(population: List[Individual]):
        Individual.sort(population)
        return population[:num]

    return elite_func_core


def tournament():
    pass
