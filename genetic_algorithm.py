from object import Circle, Object
import concurrent.futures
import numpy as np
from settings import *
import random


"""Set seed if specified"""
if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)


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

    def get_population_fitness(self, input_, output):
        # compute fitness of objects
        curr_se = np.square(np.subtract(output, input_))

        if PARALLELIZATION:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for obj in self.population:
                    executor.submit(self._get_fitness, obj, input_, output, curr_se)
        else:
            for obj in self.population:
                self._get_fitness(obj, input_, output, curr_se)

        self._sort_population()

    def _sort_population(self):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

    def _crowding(self, n, p, input_, output, curr_se):
        for _ in range(n):
            child = p.reproduce()
            crowd = random.sample(self.population, CROWD_SIZE)

            min_distance = float('inf')
            min_dist_obj = None
            for obj in crowd:
                distance = np.sqrt((obj.attr['x'] - child.attr['x']) ** 2 + (obj.attr['y'] - child.attr['y']) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    min_dist_obj = obj

            child.get_fitness(input_, output, curr_se)

            if child.fitness > min_dist_obj.fitness:
                self.population.remove(min_dist_obj)
                self.population.append(child)

    def next_gen(self, input_, output):
        curr_se = np.square(np.subtract(output, input_))

        # keep top n objects
        n = int(N_OBJECTS * SAVE_TOP_RATIO)
        parents = list(self.population)[:n]

        n_children = int(1 / SAVE_TOP_RATIO) - 1

        if PARALLELIZATION:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for p in parents:
                    executor.submit(self._crowding, n_children, p, input_, output, curr_se)
        else:
            for p in parents:
                self._crowding(n_children, p, input_, output, curr_se)
