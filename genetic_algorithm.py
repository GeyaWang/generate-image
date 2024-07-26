from object import Circle, Object
import concurrent.futures
import numpy as np
from settings import *
import random


class GeneticAlgorithm:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # create population
        self.population = [Circle(self.width, self.height) for _ in range(N_OBJECTS)]

    @staticmethod
    def _get_fitness(obj: Object, input_: np.ndarray, output: np.ndarray, curr_se: np.ndarray) -> tuple[Object, float]:
        return obj, obj.get_fitness(input_, output, curr_se)

    def get_sorted_fitness_dict(self, input_, output) -> dict[Object, float]:
        # compute fitness of objects

        curr_se = np.subtract(output, input_, dtype=np.int64) ** 2

        fit_dict = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._get_fitness, obj, input_, output, curr_se): obj for obj in self.population}

            for future in concurrent.futures.as_completed(futures):
                obj, fitness = future.result()
                fit_dict[obj] = fitness

        sorted_dict = {k: v for k, v in sorted(fit_dict.items(), key=lambda i: i[1], reverse=True)}

        return sorted_dict

    def _crossover(self, p1: Object, p2: Object) -> tuple[Object, Object]:
        crossover_point = random.randint(1, len(p1.attr) - 1)
        c1 = Circle(self.width, self.height, *(tuple(p1.attr.values())[:crossover_point] + tuple(p2.attr.values())[crossover_point:]))
        c2 = Circle(self.width, self.height, *(tuple(p2.attr.values())[:crossover_point] + tuple(p1.attr.values())[crossover_point:]))
        return c1, c2

    @staticmethod
    def _mutate(obj: Object):
        obj.mutate()

    def next_gen(self, sorted_fit_dict: dict[Object: float]):
        next_gen = []

        # keep top n objects
        n = int(N_OBJECTS * ELITISM_RATIO)
        next_gen.extend(list(sorted_fit_dict.keys())[:n])

        # tournament selection to pick parents
        parents = []
        for _ in range(N_OBJECTS - n):
            max_fitness = -float("inf")
            max_obj = None
            for _ in range(TOURNAMENT_SIZE):
                obj, fitness = random.choice(list(sorted_fit_dict.items()))
                if fitness > max_fitness:
                    max_fitness = fitness
                    max_obj = obj
            parents.append(max_obj)

        # crossover between parents
        paired_parents = []
        for i in range(len(parents) // 2):
            paired_parents.append((parents[i], parents[i + 1]))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._crossover, *p): p for p in paired_parents}

            for future in concurrent.futures.as_completed(futures):
                next_gen.extend(future.result())

        if len(parents) % 2 == 1:
            next_gen.append(parents[-1])

        # mutate next generation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for obj in next_gen:
                executor.submit(self._mutate, obj)

        self.population = next_gen
