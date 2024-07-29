from abc import ABC, abstractmethod
import numpy as np
from time import perf_counter
import cv2
import random
from settings import *


class Object(ABC):
    attr = None

    @abstractmethod
    def draw(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_fitness(self, *args, **kwargs):
        pass

    @abstractmethod
    def reproduce(self, *args, **kwargs):
        pass


class Circle(Object):
    def __init__(self, width: int, height: int, r: int = None, x: int = None, y: int = None, colour: np.ndarray = None):
        self.width = width
        self.height = height

        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.mask = None

        self.fitness = None

        if r is None:
            r = np.random.randint(0, min(width, height) // 2)
        if x is None:
            x = np.random.randint(0, width)
        if y is None:
            y = np.random.randint(0, height)
        if colour is None:
            colour = np.array((
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            ), dtype=np.int16)

        self.attr = {
            'r': r,
            'x': x,
            'y': y,
            'colour': colour,
        }

        self._set_mask()

    def _set_mask(self):
        self.min_x = max(self.attr['x'] - self.attr['r'], 0)
        self.max_x = min(self.attr['x'] + self.attr['r'], self.width)
        self.min_y = max(self.attr['y'] - self.attr['r'], 0)
        self.max_y = min(self.attr['y'] + self.attr['r'], self.height)

        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(self.mask, (self.attr['x'], self.attr['y']), self.attr['r'], 1, -1)
        self.mask = self.mask[self.min_y:self.max_y, self.min_x:self.max_x][::DOWNSAMPLING_FACTOR, ::DOWNSAMPLING_FACTOR]

    def draw(self, img_arr: np.ndarray):
        cv2.circle(
            img_arr,
            (self.attr['x'], self.attr['y']),
            self.attr['r'],
            [int(i) for i in self.attr['colour']],
            -1,
            lineType=cv2.LINE_AA
        )

    def get_fitness(self, input_img: np.ndarray, curr_img: np.ndarray, curr_se: np.ndarray):
        # crop, downsample and copy image array
        new_img = curr_img[self.min_y:self.max_y, self.min_x:self.max_x][::DOWNSAMPLING_FACTOR, ::DOWNSAMPLING_FACTOR].copy()

        # draw circle
        new_img[self.mask == 1] = self.attr['colour']

        # crop and downsample square error array
        new_se = curr_se[self.min_y:self.max_y, self.min_x:self.max_x][::DOWNSAMPLING_FACTOR, ::DOWNSAMPLING_FACTOR]

        # crop and downsample input image array
        new_input = input_img[self.min_y:self.max_y, self.min_x:self.max_x][::DOWNSAMPLING_FACTOR, ::DOWNSAMPLING_FACTOR]

        # fitness calculated as the difference of SSD with and without object, accounting for downsampling factor
        self.fitness = (
                np.sum(new_se) * DOWNSAMPLING_FACTOR ** 2 -
                np.sum(np.square(np.subtract(new_img, new_input, dtype=np.int64))) * DOWNSAMPLING_FACTOR ** 2
        )

    def reproduce(self):
        r = self.attr['r']
        x = self.attr['x']
        y = self.attr['y']
        colour_r = self.attr['colour'][0]
        colour_g = self.attr['colour'][1]
        colour_b = self.attr['colour'][2]

        if random.random() < MUTATION_CHANCE:
            r += round(r * random.uniform(-MUTATION_RATE, MUTATION_RATE))
            r = max(r, 0)
        if random.random() < MUTATION_CHANCE:
            x += int(self.width * random.uniform(-MUTATION_RATE, MUTATION_RATE))
        if random.random() < MUTATION_CHANCE:
            y += int(self.height * random.uniform(-MUTATION_RATE, MUTATION_RATE))
        if random.random() < MUTATION_CHANCE:
            colour_r += int(255 * random.uniform(-MUTATION_RATE, MUTATION_RATE))
            colour_r = min(max(colour_r, 0), 255)
        if random.random() < MUTATION_CHANCE:
            colour_g += int(255 * random.uniform(-MUTATION_RATE, MUTATION_RATE))
            colour_g = min(max(colour_g, 0), 255)
        if random.random() < MUTATION_CHANCE:
            colour_b += int(255 * random.uniform(-MUTATION_RATE, MUTATION_RATE))
            colour_b = min(max(colour_b, 0), 255)

        return Circle(self.width, self.height, r, x, y, np.array((colour_r, colour_g, colour_b), dtype=np.int16))

    def __repr__(self):
        return f'Circle(r={self.attr["r"]}, x={self.attr["x"]}, y={self.attr["y"]}, colour={self.attr["colour"]}, fit={self.fitness})'
