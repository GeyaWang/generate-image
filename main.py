import cv2
import numpy as np
import os
import glob
from model import Model
from settings import *


class GenerateImg:
    def __init__(self):
        # # init input and output img arrays
        # scale input image
        raw_input_arr = cv2.imread(IMG_PATH)
        raw_height, raw_width = raw_input_arr.shape[:2]

        self.input = cv2.cvtColor(cv2.resize(raw_input_arr, (int(raw_width * RES_SCALE), int(raw_height * RES_SCALE))), cv2.COLOR_BGR2RGB).astype(np.int64)
        self.height, self.width = self.input.shape[:2]
        self.output = np.zeros((self.height, self.width, 3), dtype=np.int64)

        # init objects
        self.genetic_model = Model(self.width, self.height)

    @staticmethod
    def _save_img(img: np.ndarray, filepath: str, verbose: bool = True):
        cv2.imwrite(filepath, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        if verbose:
            print(f'Saved file to {os.path.abspath(filepath)}')

    @staticmethod
    def _show_img(img: np.ndarray, name: str = ''):
        cv2.imshow(name, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
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

        self.genetic_model.get_population_fitness(self.input, self.output)
        for i in range(ITERATIONS):
            for _ in range(ROUNDS_PER_STEP):
                self.genetic_model.next_gen(self.input, self.output)

            if FIT_CHECK:
                while self.genetic_model.population[0].fitness < 0:
                    print('test')
                    self.genetic_model.next_gen(self.input, self.output)

            best_obj = self.genetic_model.population[0]
            print(f'\n[{i}] {best_obj=}')

            self.output = best_obj.draw(self.output)
            self._save_img(self.output, f'./temp/{i}.jpg', verbose)
            self.genetic_model.get_population_fitness(self.input, self.output)


if __name__ == '__main__':
    generate_img = GenerateImg()
    generate_img.run()
