from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import cv2
import random
from settings import *


@dataclass
class Colour:
    r: int
    g: int
    b: int

    @property
    def val(self):
        return self.r, self.g, self.b


class Object(ABC):
    attr = None

    @abstractmethod
    def draw(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_fitness(self, *args, **kwargs):
        pass

    @abstractmethod
    def mutate(self, *args, **kwargs):
        pass


class Circle(Object):
    def __init__(self, width: int, height: int, r: int = None, c_x: int = None, c_y: int = None, clr: Colour = None):
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
            clr = Colour(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        self.attr = {
            'radius': r,
            'center_x': c_x,
            'center_y': c_y,
            'colour': clr
        }

        self._set_mask()

    def _set_mask(self):
        self.min_x = max(self.attr['center_x'] - self.attr['radius'], 0)
        self.max_x = min(self.attr['center_x'] + self.attr['radius'], self.width)
        self.min_y = max(self.attr['center_y'] - self.attr['radius'], 0)
        self.max_y = min(self.attr['center_y'] + self.attr['radius'], self.height)

        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(self.mask, (self.attr['center_x'], self.attr['center_y']), self.attr['radius'], 1, -1)
        self.mask = self.mask[self.min_y:self.max_y, self.min_x:self.max_x]

    def draw(self, img_arr: np.ndarray):
        cv2.circle(
            img_arr,
            (self.attr['center_x'], self.attr['center_y']),
            self.attr['radius'],
            (self.attr['colour'].r, self.attr['colour'].g, self.attr['colour'].b),
            -1,
            lineType=cv2.LINE_AA
        )

    def get_fitness(self, input_img: np.ndarray, curr_img: np.ndarray, curr_se: np.ndarray) -> float:
        new_img_arr = curr_img[self.min_y:self.max_y, self.min_x:self.max_x].copy()
        new_img_arr[self.mask == 1] = self.attr['colour'].val

        fitness = (
                np.sum(curr_se[self.min_y:self.max_y, self.min_x:self.max_x]) -
                np.sum(np.subtract(new_img_arr, input_img[self.min_y:self.max_y, self.min_x:self.max_x], dtype=np.int64) ** 2)
        )

        return fitness

    def mutate(self):
        if random.random() < MUTATION_CHANCE:
            r_delta = round(self.attr['radius'] * MUTATION_RATE)
            c_x_delta = round(self.attr['center_x'] * MUTATION_RATE)
            c_y_delta = round(self.attr['center_y'] * MUTATION_RATE)
            clr_r_delta = round(self.attr['colour'].r * MUTATION_RATE)
            clr_g_delta = round(self.attr['colour'].g * MUTATION_RATE)
            clr_b_delta = round(self.attr['colour'].b * MUTATION_RATE)

            self.attr['radius'] += random.randint(-r_delta, r_delta)
            self.attr['center_x'] += random.randint(-c_x_delta, c_x_delta)
            self.attr['center_y'] += random.randint(-c_y_delta, c_y_delta)
            self.attr['colour'].r += random.randint(-clr_r_delta, clr_r_delta)
            self.attr['colour'].g += random.randint(-clr_g_delta, clr_g_delta)
            self.attr['colour'].b += random.randint(-clr_b_delta, clr_b_delta)

            self.attr['radius'] = max(self.attr['radius'], 0)
            self.attr['colour'].r = min(max(self.attr['colour'].r, 0), 255)
            self.attr['colour'].g = min(max(self.attr['colour'].g, 0), 255)
            self.attr['colour'].b = min(max(self.attr['colour'].b, 0), 255)

            self._set_mask()
