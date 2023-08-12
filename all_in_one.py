import numpy as np
import cv2 as cv
import time
import random

random.seed(50)


def clues_to_matrix(clues):
    clues = clues[0]+clues[1]
    output = np.zeros((40, 10), dtype=np.uint8)
    for i in range(40):
        output[i, :len(clues[i])] = clues[i]
    return output


class GridsClues:
    def __init__(self, path, dimentions=None):
        print("object created")
        self.image_path = path
        self.image_height, self.image_weight = cv.imread(self.image_path).shape[:2]
        self.dimension = dimentions if dimentions else 24
        self.remainder_values = None
        self.adding_value = 0
        self.mat20 = None
        self.mat10 = None
        self.mat5 = None
        self.mat20_clues = None
        self.mat10_clues = None
        self.mat5_clues = None
        self.flow()

    def _bin_image(self):
        print("bin image start")
        image = cv.imread(self.image_path).astype(np.uint8)
        output = np.empty(list([self.dimension]) + [self.image_height, self.image_weight], dtype=np.uint8)
        for i in range(self.image_height):
            for j in range(self.image_weight):
                temp = list("".join([format(x, 'b').zfill(8) for x in image[i, j]]))
                output[:, i, j] = [int(x) for x in temp]
        print("bin image end")
        return output

    def _hv_read(self, matrix):
        horizontal_clues = []
        vertical_clues = []
        for i in range(matrix.shape[0]):
            horizontal_clues.append(self._nono_creator(matrix[i]))
        for i in range(matrix.shape[1]):
            vertical_clues.append(self._nono_creator(matrix[:, i]))
        return horizontal_clues, vertical_clues

    def _nono_creator(self, arr):
        total_value = sum(arr)
        if total_value == 0:
            return []
        output = []
        count = 0
        total_count = 0
        for element in arr:
            if element == 1:
                count += 1
            elif count > 0:
                output.append(count)
                total_count += count
                if total_value == total_count:
                    return output
                count = 0
        if count > 0:
            output.append(count)
        return output

    def _grids_values(self):
        num = self.image_height * self.image_weight * self.dimension
        grid5 = 0
        grid10 = 0
        adding_value = 0
        grid20 = num // 400
        remainder = num % 400
        if remainder > 300:
            adding_value = 400 - remainder
            remainder = 0
            grid20 += 1
        else:
            grid10 = remainder // 100
            remainder = remainder % 100
            if remainder > 60:
                adding_value = 100 - remainder
                remainder = 0
                grid10 += 1
            else:
                grid5 = remainder // 25
                remainder = remainder % 25
                if remainder > 15:
                    adding_value = 25 - remainder
                    remainder = 0
                    grid5 += 1

        self.num_mat20, self.num_mat10, self.num_mat5, self.adding_value = grid20, grid10, grid5, adding_value
        return np.cumsum([grid20 * 400, grid10 * 100, grid5 * 25]), num - remainder

    def flow(self):
        start = time.time()
        print("flow start time:", start)
        bin_out = self._bin_image().reshape(-1)
        print("bin take", time.time())
        values, output_range = self._grids_values()
        if self.adding_value:
            bin_out = np.concatenate((bin_out, np.ones(self.adding_value, dtype=np.uint8)))
        elif output_range:
            self.remainder_values = bin_out[output_range:]
            bin_out = bin_out[:output_range]

        mat20 = bin_out[:values[0]].reshape(-1, 20, 20)
        ############################
        index = []
        print("indexing start time:", time.time())
        for i in range(mat20.shape[0]):
            if np.sum(mat20[i]) > 0:
                index.append(i)
        print("indexing endtime:", time.time())
        mat20 = mat20[index]
        print("empty matrix removed time:", time.time())
        clues = []
        print("total matrix:",mat20.shape[0])
        for i, x in enumerate(mat20):
            if i % 1000 == 0:
                print(f"{i/1000}k mat done")
            clues.append(self._hv_read(x))
        print("hv clues created time:", time.time())
        clues = np.array([clues_to_matrix(x) for x in clues])
        print("hv clues matrix created time:", time.time())
        mat20 = mat20.reshape(-1, 400)
        print("matrix is reshaped time:", time.time())
        clues = clues.reshape(-1, 400)
        print("clues is reshaped time:", time.time())
        with open(r"E:\data\matrix.csv", "ab") as m:
            np.savetxt(m, mat20, delimiter=",", fmt="%1u")
        print("matrix is saved time:", time.time())
        with open(r"E:\data\clues.csv", "ab") as c:
            np.savetxt(c, clues, delimiter=",", fmt="%2u")
        print("clues is saved time:", time.time())

        print("flow end time:", time.time()-start)
        ############################
        # self.mat10 = bin_out[values[0]:values[1]].reshape(-1, 10, 10)
        # self.mat5 = bin_out[values[1]:values[2]].reshape(-1, 5, 5)

        ############################
        ############################
        # self.mat20_clues = [self._hv_read(x) for x in self.mat20]
        # self.mat10_clues = [self._hv_read(x) for x in self.mat10]
        # self.mat5_clues = [self._hv_read(x) for x in self.mat5]



if __name__ == '__main__':
    # obj1 = GridsClues("picture.jpg")
    # obj2 = GridsClues("picture2.jpg")
    obj3 = GridsClues("picture1.jpg")


