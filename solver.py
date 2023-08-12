"""
Given a solution for a nonogram puzzle:
1. prove it can be solvable
2. if it cannot be solvable, add constraints until the solution is unique
"""

import numpy as np
import cv2 as cv
import json
import time

# import initiator
# import mediator
# import solver


import random

random.seed(50)


def solve(
        nonogram, starter=None, add_puzzle_constraints=False):

    nonogram_solver = NonogramSolver(nonogram)

    if starter is not None:
        for i in range(starter):
            nonogram_solver._pick_help_square()

    nonogram_solver._generate_solutions()

    if add_puzzle_constraints:
        help = 1
        while not nonogram_solver._puzzle_is_solved():
            # pick a random filled in square
            # generate solutions until the puzzle is solved
            print("get help: ", help)
            help += 1
            nonogram_solver._pick_help_square()

            nonogram_solver._generate_solutions()

    return nonogram_solver._puzzle_is_solved(), nonogram_solver


def possibilities_generator(
        prior, min_pos, max_start_pos, constraint_len, total_filled):

    prior_filled = np.zeros(len(prior)).astype(bool)
    prior_filled[prior == 1] = True
    prior_empty = np.zeros(len(prior)).astype(bool)
    prior_empty[prior == 0] = True
    for start_pos in range(min_pos, max_start_pos + 1):
        possible = -1 * np.ones(len(prior))
        possible[start_pos:start_pos + constraint_len] = 1
        if start_pos + constraint_len < len(possible):
            possible[start_pos + constraint_len] = 0
        if start_pos > 0:
            possible[start_pos - 1] = 0

        # add in the prior
        possible[np.logical_and(possible == -1, prior == 0)] = 0
        possible[np.logical_and(possible == -1, prior == 1)] = 1

        # if contradiction with prior, continue
        # 1. possible changes prior = 1 to something else
        # 2. possible changes prior = 0 to something else
        # 3. everything is assigned in possible but there are not
        #    enough filled in
        # 4. possible changes nothing about the prior
        if np.any(possible[np.where(prior == 1)[0]] != 1) or \
                np.any(possible[np.where(prior == 0)[0]] != 0) or \
                np.sum(possible == 1) > total_filled or \
                (np.all(possible >= 0) and np.sum(possible == 1) <
                 total_filled) or \
                np.all(prior == possible):
            continue
        yield possible


def generate_constraints_line(constraints, prior, size):
    """
    example:
    row length = 4, row constraint = 2 (it is the only constraint)
    min_pos = 0, max_pos = 3, max_start_pos = 2
    row length = 5, row constraints 1, 2
    constraint 1:
        min_pos = 0, max_pos = 1, max_start_pos = 1
    constraint 2:
        min_pos = 2, max_pos = 4, max_start_pos = 3
    """
    global index
    if len(constraints) == 0:
        return np.logical_not(np.ones(len(prior))), np.logical_not(np.zeros(len(prior)))

    total_filled = np.sum(constraints)

    # if the number of filled in squares is equal to the total
    # constraints, fill in the rest with empties and call it a day
    if np.sum(prior == 1) == total_filled:
        for index in range(len(prior)):
            if prior[index] == -1:
                prior[index] = 0
        return (prior == 1, prior == 0)

    # if there is only one square left un-solved, solve it
    index_unknown = np.where(prior == -1)[0]
    if len(index_unknown) == 1:
        currently_filled_in_prior = np.sum(prior == 1)
        if total_filled == currently_filled_in_prior:
            prior[index_unknown[0]] = 0
        else:
            prior[index_unknown[0]] = 1

        return (prior == 1, prior == 0)

    # if any constraints are solved, don't
    # iterate over them
    indices_to_remove = []
    constraint_indices = np.ones(len(constraints))

    start_blocks = []
    start_would_be_zero_index = []
    block = 0
    for index, item in zip(list(range(len(prior))), prior):
        if item != -1 and item == 1:
            block += 1
        else:
            if block > 0:
                start_would_be_zero_index.append(index)
                start_blocks.append(block)
                block = 0
            if item == -1:
                break
    if block > 0:
        start_would_be_zero_index.append(index)
        start_blocks.append(block)

    index = 0
    for block, constraint in zip(
            start_blocks, constraints[:len(start_blocks)]):
        if block == constraint:
            # remove from iteration
            indices_to_remove.append(index)
            # if the block is correct, update the prior appropriately
            prior[start_would_be_zero_index[index]] = 0
            before_index = start_would_be_zero_index[index] - constraint - 1
            if before_index > 0:
                prior[before_index] = 0
        else:
            break
        index += 1

    end_blocks = []
    end_would_be_zero_index = []
    reverse_index_range = list(range(len(prior)))
    reverse_index_range = reversed(reverse_index_range)
    block = 0
    for index, item in zip(reverse_index_range, prior[::-1]):
        if item != -1 and item == 1:
            block += 1
        else:
            if block > 0:
                end_would_be_zero_index.append(index)
                end_blocks.append(block)
                block = 0
            if item == -1:
                break
    if block > 0:
        end_would_be_zero_index.append(index)
        end_blocks.append(block)

    index = len(constraints) - 1
    for block, constraint in zip(
            end_blocks[::-1],
            constraints[-1 * len(end_blocks):]):
        if block == constraint:
            # remove from iteration
            indices_to_remove.append(index)
            # if the block is correct, update the prior appropriately
            prior[end_would_be_zero_index[len(constraints) - 1 - index]] = 0
            before_index = end_would_be_zero_index[
                               len(constraints) - 1 - index] + constraint + 1
            if before_index > 0 and before_index < len(prior):
                prior[before_index] = 0
        else:
            break
        index -= 1

    for index in indices_to_remove:
        constraint_indices[index] = 0

    # if all the constraints are met, make sure the
    # -1s are 0
    if np.all(constraint_indices == 0):
        for index in range(len(prior)):
            if prior[index] == -1:
                prior[index] = 0
        return (prior == 1, prior == 0)

    possibilities_filled = prior == 1
    possibilities_empty = prior == 0

    for constraint_index, constraint in zip(
            list(range(len(constraints))), constraints):
        if constraint_indices[constraint_index] == 0:
            continue

        possibilities_filled_temp = np.ones(size).astype(bool)
        possibilities_empty_temp = np.ones(size).astype(bool)
        min_pos = sum(constraints[:constraint_index]) + constraint_index
        max_pos = size - 1 - (
                sum(constraints[constraint_index + 1:]) +
                len(constraints[constraint_index + 1:]))

        # if we already have everything in that window
        # just continue
        if np.all(prior[min_pos:max_pos + 1] != -1):
            possibilities_filled[min_pos:max_pos + 1] = \
                prior[min_pos:max_pos + 1] == 1
            possibilities_empty[min_pos:max_pos + 1] = \
                prior[min_pos:max_pos + 1] == 0
            continue

        # if you have some contiguous thing from the prior in the range of
        # min_pos and max_pos, fix it
        potential_indices = np.where(prior[min_pos:max_pos + 1] != 0)[0]
        if len(potential_indices) > 0:
            potential_indices += min_pos
            min_pos = min(potential_indices)
            max_pos = max(potential_indices)

        max_start_pos = max_pos + 1 - constraint

        # the smart logic about the constraints is generated
        # in the possibilities_generator
        i = 0
        for possible_nums in possibilities_generator(
                prior, min_pos, max_start_pos, constraint, total_filled):
            possible_filled = possible_nums == 1
            possible_empty = possible_nums != 1
            possibilities_filled_temp = np.logical_and(
                possibilities_filled_temp, possible_filled)
            possibilities_empty_temp = np.logical_and(
                possibilities_empty_temp, possible_empty)
            i += 1

        if i > 0:
            possibilities_filled[min_pos:max_pos + 1] = \
                possibilities_filled_temp[min_pos:max_pos + 1]
            possibilities_empty[min_pos:max_pos + 1] = \
                possibilities_empty_temp[min_pos:max_pos + 1]

        # TODO: you can probably end early here. figure out the constraint
        # if np.all(possibilities_filled == False) and \
        #         np.all(possibilities_empty == False):
        #     break
    return (possibilities_filled, possibilities_empty)

def generate_solutions(
        n_rows, n_cols, rows_constraints, cols_constraints, puzzle):

    n_changed = 1
    iters = 0
    while n_changed > 0:
        n_changed = 0
        for row_index, row_constraints in zip(list(range(n_rows)), rows_constraints):
            # if the row has no empty spaces left,
            # just skip that set of constraints
            if np.sum(puzzle[row_index, :] == -1) == 0:
                continue

            (possibilities_filled, possibilities_empty) = \
                generate_constraints_line(
                    row_constraints,
                    puzzle[row_index, :].copy(),
                    n_cols)

            for index in range(len(possibilities_filled)):
                if possibilities_filled[index] and \
                        puzzle[row_index, index] == 1:
                    possibilities_filled[index] = False

            for index in range(len(possibilities_empty)):
                if possibilities_empty[index] and \
                        puzzle[row_index, index] == 0:
                    possibilities_empty[index] = False

            if np.all(np.logical_not(possibilities_filled)) and \
                    np.all(np.logical_not(possibilities_empty)):
                continue

            puzzle[row_index, :][possibilities_filled] = 1
            puzzle[row_index, :][possibilities_empty] = 0

            n_changed += sum(possibilities_filled)
            n_changed += sum(possibilities_empty)

        print("n_changed = %s" % n_changed)

        # cols
        for col_index, col_constraints in zip(list(range(n_cols)), cols_constraints):
            # if the row has no empty spaces left,
            # just skip that set of constraints
            if np.sum(puzzle[:, col_index] == -1) == 0:
                continue

            (possibilities_filled, possibilities_empty) = \
                generate_constraints_line(
                    col_constraints,
                    puzzle[:, col_index].copy(),
                    n_rows)

            for index in range(len(possibilities_filled)):
                if possibilities_filled[index] and \
                        puzzle[index, col_index] == 1:
                    possibilities_filled[index] = False

            for index in range(len(possibilities_empty)):
                if possibilities_empty[index] and \
                        puzzle[index, col_index] == 0:
                    possibilities_empty[index] = False

            if np.all(np.logical_not(possibilities_filled)) and \
                    np.all(np.logical_not(possibilities_empty)):
                continue

            puzzle[:, col_index][possibilities_filled] = 1
            puzzle[:, col_index][possibilities_empty] = 0

            n_changed += sum(possibilities_filled)
            n_changed += sum(possibilities_empty)

        print("n_changed = %s" % n_changed)
        print("finished iter %s" % iters)

        iters += 1
    return puzzle


class NonogramSolver(object):
    def __init__(self, nonogram):
        self.nonogram = nonogram
        self.puzzle_state = -1 * np.ones((nonogram.n_rows, nonogram.n_cols))
        self.filled_positions_hint_eligible = nonogram.solution_list
        self.prefilled_positions = []

    def _generate_solutions(self):
        self.puzzle_state = generate_solutions(
            self.nonogram.n_rows, self.nonogram.n_cols,
            self.nonogram.rows_constraints,
            self.nonogram.cols_constraints,
            self.puzzle_state)

        # update the positions that can be used for hints appropriately
        for row_index, row in zip(list(range(
                self.nonogram.n_rows)), self.puzzle_state):
            for col_index, col_elem in zip(list(range(self.nonogram.n_cols)), row):
                pos = (row_index, col_index)
                if col_elem == 1 and \
                        pos in self.filled_positions_hint_eligible:
                    self.filled_positions_hint_eligible.remove(pos)

    def _pick_help_square(self, position=None):
        # randomly pick an element in the filled list
        # add that filled in thing to the puzzle solution
        print("%s squares available" % (
            len(self.filled_positions_hint_eligible)))
        if len(self.filled_positions_hint_eligible) == 0:
            raise ValueError("No more positions available")

        if position is None:
            filled_square = random.choice(self.filled_positions_hint_eligible)
        else:
            filled_square = position

        self.filled_positions_hint_eligible.remove(filled_square)
        self.prefilled_positions.append(filled_square)

        self.puzzle_state[filled_square[0], filled_square[1]] = 1

    def _puzzle_is_solved(self):
        if np.sum(self.puzzle_state == -1) == 0:
            return True
        return False


# if __name__ == '__main__':
#     img = cv.imread("picture.jpg")
#     matrix = initiator.last_matrix(img)
#     solvable_matrix = matrix[:,:,20,13,0,0]
#     unsolvable_matrix = matrix[:,:,24,0,0,0]
#     for mat in [solvable_matrix,unsolvable_matrix]:
#         clues = mediator.Mediator(mat)
#         nonogram = solver.NonogramSolver(clues)
#         nonogram._generate_solutions()
#         solvable = nonogram._puzzle_is_solved()
