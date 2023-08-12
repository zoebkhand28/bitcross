import numpy as np
from solver import solve



def solution(matrix):
    rows, columns = np.where(matrix == 1)
    output = []
    for index in range(len(rows)):
        output.append([rows[index], columns[index]])
    return output


def hv_read(matrix):
    height, width = matrix.shape
    horizontal_read = [[[matrix[j, i]] for i in range(width)] for j in range(height)]
    vertical_read = [[[matrix[j, i]] for j in range(height)] for i in range(width)]
    horizontal_clues = list(map(_nono_creator_, horizontal_read))
    vertical_clues = list(map(_nono_creator_, vertical_read))

    return horizontal_clues, vertical_clues

def _nono_creator_(inp):
    arr = inp
    i = 0
    clues = []
    count = 0
    if len(arr) == 0:
        return []
    curr = arr[0]
    while i < len(arr):
        while i < len(arr) and str(arr[i]) == str(curr):
            i += 1
            count += 1
        clues.append((curr, count))
        if i < len(arr):
            count = 0
            curr = arr[i]
    output = []
    for element in clues:
        if element[0][0] == 1:
            output.append(element[1])
    return output


class Mediator(object):
    def __init__(self, matrix):
        self.n_rows, self.n_cols = matrix.shape
        self.rows_constraints, self.cols_constraints = hv_read(matrix)
        self.solution_list = solution(matrix)


if __name__ == '__main__':
    img = [[1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
           [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
           [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
           [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
           [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
           [1, 0, 0, 0, 1, 1, 0, 0, 0, 1]]
    img = np.array(img)
    nonogram = Mediator(img)
    print(img.shape)
    print(nonogram.rows_constraints,nonogram.cols_constraints)
