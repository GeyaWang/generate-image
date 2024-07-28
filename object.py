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
    def __init__(self, width: int, height: int, r: int = None, c_x: int = None, c_y: int = None, clr: np.ndarray = None, a: float = None):
        self.width = width
        self.height = height

        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.mask = None

        if r is None:
            r = np.random.randint(0, min(width, height) // 2)
        if c_x is None:
            c_x = np.random.randint(0, width)
        if c_y is None:
            c_y = np.random.randint(0, height)
        if clr is None:
            # clr = np.array((
            #     np.random.randint(0, 255),
            #     np.random.randint(0, 255),
            #     np.random.randint(0, 255),
            # ), dtype=np.int16)

            clr = np.array((
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            ), dtype=np.int16)
        if a is None:
            a = np.random.random()

        self.attr = {
            'radius': r,
            'center_x': c_x,
            'center_y': c_y,
            'colour': clr,
            'alpha': a
        }

        self._set_mask()

    def _set_mask(self):
        self.min_x = max(self.attr['center_x'] - self.attr['radius'], 0)
        self.max_x = min(self.attr['center_x'] + self.attr['radius'], self.width)
        self.min_y = max(self.attr['center_y'] - self.attr['radius'], 0)
        self.max_y = min(self.attr['center_y'] + self.attr['radius'], self.height)

        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(self.mask, (self.attr['center_x'], self.attr['center_y']), self.attr['radius'], 1, -1)
        self.mask = self.mask[self.min_y:self.max_y, self.min_x:self.max_x][::DOWNSAMPLING_FACTOR, ::DOWNSAMPLING_FACTOR].astype(bool)

    def draw(self, img_arr: np.ndarray) -> np.ndarray:
        overlay = img_arr.copy()
        cv2.circle(
            overlay,
            (self.attr['center_x'], self.attr['center_y']),
            self.attr['radius'],
            [int(i) for i in self.attr['colour']],
            -1,
            lineType=cv2.LINE_AA
        )
        img_arr = cv2.addWeighted(overlay, self.attr['alpha'], img_arr, 1 - self.attr['alpha'], 0)
        return img_arr

    def get_fitness(self, input_img: np.ndarray, curr_img: np.ndarray, curr_se: np.ndarray) -> float:
        # crop, downsample and copy image array
        new_img = curr_img[self.min_y:self.max_y, self.min_x:self.max_x][::DOWNSAMPLING_FACTOR, ::DOWNSAMPLING_FACTOR].copy()

        # draw circle using alpha channel
        new_img[self.mask] = (1 - self.attr['alpha']) * new_img[self.mask] + self.attr['alpha'] * self.attr['colour']

        # crop and downsample square error array
        new_se = curr_se[self.min_y:self.max_y, self.min_x:self.max_x][::DOWNSAMPLING_FACTOR, ::DOWNSAMPLING_FACTOR]

        # crop and downsample input image array
        new_input = input_img[self.min_y:self.max_y, self.min_x:self.max_x][::DOWNSAMPLING_FACTOR, ::DOWNSAMPLING_FACTOR]

        # fitness calculated as the difference of SSD with and without object, accounting for downsampling factor
        fitness = (
                np.sum(new_se) * DOWNSAMPLING_FACTOR ** 2 -
                np.sum(np.square(np.subtract(new_img, new_input, dtype=np.int64))) * DOWNSAMPLING_FACTOR ** 2
        )

        return fitness

    def reproduce(self):
        if random.random() < MUTATION_CHANCE:
            r = round(self.attr['radius'] * (1 + random.uniform(-MUTATION_RATE, MUTATION_RATE)))
            c_x = round(self.attr['center_x'] * (1 + random.uniform(-MUTATION_RATE, MUTATION_RATE)))
            c_y = round(self.attr['center_y'] * (1 + random.uniform(-MUTATION_RATE, MUTATION_RATE)))
            clr_r = round(self.attr['colour'][0] * (1 + random.uniform(-MUTATION_RATE, MUTATION_RATE)))
            clr_g = round(self.attr['colour'][1] * (1 + random.uniform(-MUTATION_RATE, MUTATION_RATE)))
            clr_b = round(self.attr['colour'][2] * (1 + random.uniform(-MUTATION_RATE, MUTATION_RATE)))
            a = self.attr['alpha'] * (1 + random.uniform(-MUTATION_RATE, MUTATION_RATE))

            r = max(self.attr['radius'], 0)
            clr_r = min(max(self.attr['colour'][0], 0), 255)
            clr_g = min(max(self.attr['colour'][1], 0), 255)
            clr_b = min(max(self.attr['colour'][2], 0), 255)
            a = min(max(self.attr['alpha'], 0), 1)
        else:
            r = self.attr['radius']
            c_x = self.attr['center_x']
            c_y = self.attr['center_y']
            clr_r = self.attr['colour'][0]
            clr_g = self.attr['colour'][1]
            clr_b = self.attr['colour'][2]
            a = self.attr['alpha']

        return Circle(self.width, self.height, r, c_x, c_y, np.array((clr_r, clr_g, clr_b), dtype=np.int16), a)

    def __repr__(self):
        return f'Circle(width={self.width}, height={self.height}, r={self.attr['radius']}, c_x={self.attr['center_x']}, c_y={self.attr['center_y']}, clr={self.attr['colour']}, a={self.attr['alpha']})'
