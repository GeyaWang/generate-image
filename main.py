import cv2
import numpy as np
import os
import glob
from genetic_algorithm import GeneticAlgorithm
from settings import *
import random
from helper import timer


class GenerateImg:
    def __init__(self):
        # # init input and output img arrays
        # scale input image
        raw_input_arr = cv2.imread(IMG_PATH)
        raw_height, raw_width = raw_input_arr.shape[:2]

        self.input = cv2.cvtColor(cv2.resize(raw_input_arr, (int(raw_width * RES_SCALE), int(raw_height * RES_SCALE))), cv2.COLOR_BGR2RGB)
        self.height, self.width = self.input.shape[:2]
        self.output = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # init objects
        self.genetic_agent = GeneticAlgorithm(self.width, self.height)

    @staticmethod
    def _save_img(img: np.ndarray, filepath: str, verbose: bool = True):
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if verbose:
            print(f'Saved file to {os.path.abspath(filepath)}')

    @staticmethod
    def _show_img(img: np.ndarray, name: str = ''):
        cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self, verbose: bool = True):
        if not os.path.exists('./temp'):
            # create temp folder
            os.makedirs('./temp')
            print(f'Created directory {os.path.abspath('./temp')}')
        else:
            # empty temp folder
            files = glob.glob('./temp/*')
            if files:
                print(f'Emptied directory {os.path.abspath('./temp')}')
            for f in files:
                os.remove(f)

        for i in range(ITERATIONS):

            sorted_fit_dict = None
            for _ in range(ROUNDS_PER_STEP):
                # compute fitness of objects
                fit_time, sorted_fit_dict = timer(self.genetic_agent.get_sorted_fitness_dict, (self.input, self.output))

                # get next generation
                self.genetic_agent.next_gen(sorted_fit_dict)

            # get the best fitness and object
            max_obj, max_fitness = list(sorted_fit_dict.items())[0]

            self.output = max_obj.draw(self.output)
            self._save_img(self.output, f'./temp/{i}.jpg')

            if verbose:
                print(f'{max_obj=}')
                print(f'[{i}] max_fitness={int(max_fitness)}, fitness_time={fit_time} ({fit_time / N_OBJECTS} per), population_size={len(self.genetic_agent.population)}')
        #
        # self._show_img(self.output)


if __name__ == '__main__':
    generate_img = GenerateImg()
    generate_img.run()
