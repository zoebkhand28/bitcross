import numpy as np
from solver import possibilities_generator


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
    prior = np.array(prior)
    # global index
    if len(constraints) == 0:
        return np.zeros(size, dtype=bool), np.ones(size, dtype=bool)

    total_filled = np.sum(constraints)

    # if the number of filled in squares is equal to the total
    # constraints, fill in the rest with empties and call it a day
    if np.sum(prior == 1) == total_filled:
        prior = np.where(prior == -1, 0, prior)
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
    for index, item in enumerate(prior):
        if item == 1:
            block += 1
        else:
            if block > 0:
                start_would_be_zero_index.append(index)
                start_blocks.append(block)
                block = 0
            if item == -1:
                break


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
    reverse_index_range = reversed(list(range(len(prior))))
    block = 0
    for index, item in zip(reverse_index_range, prior[::-1]):
        if item == 1:
            block += 1
        else:
            if block > 0:
                end_would_be_zero_index.append(index)
                end_blocks.append(block)
                block = 0
            if item == -1:
                break

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
            if before_index > 0 and before_index < size:
                prior[before_index] = 0
        else:
            break
        index -= 1

    for index in indices_to_remove:
        constraint_indices[index] = 0

    # if all the constraints are met, make sure the
    # -1s are 0
    if np.all(constraint_indices == 0):
        for index in range(size):
            if prior[index] == -1:
                prior[index] = 0
        return (prior == 1, prior == 0)
    possibilities_filled = prior == 1
    possibilities_empty = prior == 0

    for constraint_index, constraint in enumerate(constraints):
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
            possibilities_filled[min_pos:max_pos + 1] = prior[min_pos:max_pos + 1] == 1
            possibilities_empty[min_pos:max_pos + 1] = prior[min_pos:max_pos + 1] == 0
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
    return possibilities_filled, possibilities_empty

if __name__=="__main__":
    constraints = np.array([3,4])
    n_col = 10
    prior = -1 * np.ones(10)
    prior[:3] = 1
    prior[3:5] = 0
    prior[5:8] = 1

    output = generate_constraints_line(constraints, prior, n_col)
    print(output[0])
    print(output[1])

