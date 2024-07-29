import cv2
import numpy as np
import os
import glob
from genetic_algorithm import GeneticAlgorithm
from settings import *
import time


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

    @ staticmethod
    def _create_temp_dir():
        if not os.path.exists('./temp'):
            # create temp folder
            os.makedirs('./temp')
            print(f'Created directory {os.path.abspath("./temp")}')
        else:
            # empty temp folder
            files = glob.glob('./temp/*')
            if files:
                print(f'Emptied directory {os.path.abspath("./temp")}')
            for f in files:
                os.remove(f)

    def run(self, verbose: bool = True):
        self._create_temp_dir()

        gen_time = None
        for i in range(ITERATIONS):

            t1 = time.perf_counter()
            for _ in range(ROUNDS_PER_STEP):
                self.genetic_agent.get_population_fitness(self.input, self.output)
            fit_time = time.perf_counter() - t1

            best_obj = self.genetic_agent.population[0]
            best_obj.draw(self.output)

            if verbose:
                print(f'[{i}] max_fitness={int(best_obj.fitness)}, population_size={len(self.genetic_agent.population)}, {fit_time=}, {gen_time=}')
                print(f'{best_obj=}')
            self._save_img(self.output, f'./temp/{i}.jpg', verbose)

            t1 = time.perf_counter()
            self.genetic_agent.next_gen(self.input, self.output)
            gen_time = time.perf_counter() - t1


if __name__ == '__main__':
    generate_img = GenerateImg()
    generate_img.run()
