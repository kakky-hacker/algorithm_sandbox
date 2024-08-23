import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import random

from core import MuCommaLambdaGA, MuPlusLambdaGA
from crossover import one_point
from mutation import random_int
from selection import elite

prices = [100, 200, 650, 1100, 2100, 3300]
values = [2, 5, 16, 32, 64, 105]
money = 40000


def generate_func():
    ans = [0] * 6
    for _ in range(200):
        i = random.randint(0, 5)
        ans[i] += 1
        if objective_func(ans) == 0:
            ans[i] -= 1
            return ans
    ans[random.randint(0, 5)] = 0
    return ans


def objective_func(ans):
    total_value = 0
    _money = money
    for i, n in enumerate(ans):
        _money -= prices[i] * n
        total_value += values[i] * n
    if _money < 0:
        total_value = 0
    return [total_value]


selction_func = elite(100)
crossover_func = one_point()
mutation_func = random_int(0, 100, 0.3, 0.4)

ga = MuPlusLambdaGA(
    "maximize",
    objective_func,
    generate_func,
    selction_func,
    crossover_func,
    mutation_func,
)
ans = ga.solve(n_generations=1000, population_size=200)
ans.sort(key=lambda x: x.objective, reverse=True)
print(ans[0].objective, ans[0].chromosome)
