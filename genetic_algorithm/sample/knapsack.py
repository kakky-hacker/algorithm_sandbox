import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import random

from core import GeneticAlgorithm
from crossover import one_point
from mutation import random_int
from selection import elite


def generate_func():
    ans = [0] * 6
    for _ in range(300):
        i = random.randint(0, 5)
        ans[i] += 1
        if objective_func(ans) == 0:
            ans[i] -= 1
            return ans
    return ans


def objective_func(ans):
    prices = [100, 200, 300, 650, 1100, 1500]
    values = [2, 5, 8, 16, 32, 46]
    money = 30000
    total_value = 0
    for i, n in enumerate(ans):
        money -= prices[i] * n
        total_value += values[i] * n
    if money < 0:
        total_value = 0
    return [total_value]


selction_func = elite(30)
crossover_func = one_point()
mutation_func = random_int(1, 100, 0.3, 0.4)

ga = GeneticAlgorithm(
    "maximize",
    objective_func,
    generate_func,
    selction_func,
    crossover_func,
    mutation_func,
)
ans = ga.solve(n_generations=5000, n_population=100)[0]
print(ans.objective, ans.chromosome)
