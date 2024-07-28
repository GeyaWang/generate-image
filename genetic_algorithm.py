from object import Circle, Object
import concurrent.futures
import numpy as np
from helper import sort_dict
from settings import *
import random


class GeneticAlgorithm:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # create population
        self.population = [Circle(self.width, self.height) for _ in range(N_OBJECTS)]
        self.children = []

    @staticmethod
    def _get_fitness(obj: Object, input_: np.ndarray, output: np.ndarray, curr_se: np.ndarray):
        obj.get_fitness(input_, output, curr_se)

    def get_population_fitness(self, input_, output) -> dict[Object, float]:
        # compute fitness of objects
        curr_se = np.subtract(output, input_, dtype=np.int64) ** 2
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for obj in self.population:
                executor.submit(self._get_fitness, obj, input_, output, curr_se)

        self._sort_population()

    def _sort_population(self):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

    @staticmethod
    def _get_children(n: int, obj: Object):
        children = []
        for _ in range(n):
            children.append(obj.reproduce())
        return children

    def _get_next_gen(self, sorted_fit_dict: dict[Object: float]):
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

        return next_gen

    def next_gen(self, input_, output):
        curr_se = np.subtract(output, input_, dtype=np.int64) ** 2

        # keep top n objects
        n = int(N_OBJECTS * SAVE_TOP_RATIO)
        parents = list(self.population)[:n]

        n_children = int(1 / SAVE_TOP_RATIO) - 1
        for p in parents:
            for _ in range(n_children):
                child = p.reproduce()
                crowd = random.sample(self.population, CROWD_SIZE)

                min_distance = float('inf')
                min_dist_obj = None
                min_dist_fit = None
                for obj in crowd:
                    distance = np.sqrt((obj.attr['x'] - child.attr['x']) ** 2 + (obj.attr['y'] - child.attr['y']) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        min_dist_obj = obj

                child.get_fitness(input_, output, curr_se)

                if child.fitness > min_dist_obj.fitness:
                    self.population.remove(min_dist_obj)
                    self.population.append(child)

    def play_step(self, input_, output):
        self.get_population_fitness(input_, output)

        best_obj = self.population[0]
        best_obj.draw(input_)

        self.next_gen(input_, output)
