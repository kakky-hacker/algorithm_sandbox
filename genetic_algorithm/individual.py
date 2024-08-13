class Individual:
    def __init__(self, chromosome, objective, direction):
        assert direction in ("minimize", "maximize")
        self.chromosome = chromosome
        self.objective = objective
        self.direction = direction

    def superior(self, other) -> bool:
        return all(
            [
                b <= a if self.direction == "maximize" else a <= b
                for a, b in zip(self.objective, other.objective)
            ]
        )

    @classmethod
    def instantiate(self, generate_func, objective_func, direction):
        chromosome = generate_func()
        objective = objective_func(chromosome)
        return Individual(chromosome, objective, direction)

    @classmethod
    def sort(self, population):
        for i in range(len(population) - 1):
            for j in range(len(population) - 1, i, -1):
                if population[j].superior(population[j - 1]):
                    population[j], population[j - 1] = (
                        population[j - 1],
                        population[j],
                    )
