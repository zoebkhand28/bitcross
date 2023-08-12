import numpy as np
from numba import njit
import cv2 as cv
import pickle
import time

import initiator
import mediator
import solver



def pipeline(image):
    solvable_matrix = 0
    not_solvable_matrix = 0
    start_time = time.time()
    matrix = initiator.last_matrix(image)
    x, y, z, p = matrix.shape[2:]
    dictionary_pixel = {}
    dictionary_z = {}
    dictionary_y = {}
    dictionary_x = {}
    pixel = 0
    while pixel < p:
        z_axis = 0
        while z_axis < z:
            y_axis = 0
            while y_axis < y:
                x_axis = 0
                while x_axis < x:
                    filled_places = np.sum(matrix[:, :, x_axis, y_axis, z_axis, pixel])
                    if filled_places > 10:
                        clues = mediator.Mediator(matrix[:, :, x_axis, y_axis, z_axis, pixel])
                        nonogram = solver.NonogramSolver(clues)
                        nonogram._generate_solutions()
                        solvable = nonogram._puzzle_is_solved()
                        if solvable:
                            solvable_matrix += 1
                            dictionary_x[x_axis] = {"constraints": [clues.rows_constraints, clues.cols_constraints]}
                        else:
                            not_solvable_matrix += 1
                            while not nonogram._puzzle_is_solved():
                                nonogram._pick_help_square()
                                nonogram._generate_solutions()
                            dictionary_x[x_axis] = {"constraints": [clues.rows_constraints, clues.cols_constraints],
                                                    "positions": nonogram.prefilled_positions}
                    elif 10 >= filled_places > 0:
                        clues = mediator.Mediator(matrix[:, :, x_axis, y_axis, z_axis, pixel])
                        dictionary_x[x_axis] = {"solutions": clues.solution_list}
                    x_axis += 1
                if len(dictionary_x.keys()) > 0:
                    dictionary_y[y_axis] = dictionary_x
                    dictionary_x = {}
                y_axis += 1
            if len(dictionary_y.keys()) > 0:
                dictionary_z[z_axis] = dictionary_y
                dictionary_y = {}
            z_axis += 1
        if len(dictionary_z.keys()) > 0:
            dictionary_pixel[pixel] = dictionary_z
            dictionary_z = {}
        pixel += 1
    print("solvable_matrix",solvable_matrix,"not_solvable_matrix",not_solvable_matrix)
    return time.time()-start_time, dictionary_pixel


if __name__ == '__main__':
    img = cv.imread("picture.jpg")
    ttime, output = pipeline(img)
    print(f'time: {ttime // 60}min {ttime % 60}sec')
    with open("sample.pkl", "wb") as outfile:
        pickle.dump(output, outfile)
        outfile.close()