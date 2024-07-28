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

        # fit_dict = {obj: self._get_fitness(obj, input_, output, curr_se)[1] for obj in self.population}
        # print(fit_dict)

        sorted_dict = {k: v for k, v in sorted(fit_dict.items(), key=lambda i: i[1], reverse=True)}

        return sorted_dict

    @staticmethod
    def _get_children(n: int, obj: Object):
        children = []
        for _ in range(n):
            children.append(obj.reproduce())
        return children

    def next_gen(self, sorted_fit_dict: dict[Object: float]):
        next_gen = []

        # keep top n objects
        n = int(N_OBJECTS * SAVE_TOP_RATIO)
        parents = list(sorted_fit_dict.keys())[:n]
        next_gen.extend(parents)

        n_children = int(1 / SAVE_TOP_RATIO) - 1
        # for p in parents:
        #     for _ in range(n_children):
        #         child = p.reproduce()
        #         next_gen.append(child)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._get_children, n_children, p): p for p in parents}

            for future in concurrent.futures.as_completed(futures):
                children = future.result()
                next_gen.extend(children)

        self.population = next_gen
