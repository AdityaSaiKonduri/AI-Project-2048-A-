import random
from copy import deepcopy
import math
# import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

random_next_states = []
def random_generate_2(grid):
    while True:
        x = random.choice([0, 1, 2, 3])
        y = random.choice([0, 1, 2, 3])
        m = random.choice([2, 4, 8])
        if grid[x][y] == 0:
            grid[x][y] = m
            return grid

def merge_tiles(row):
    for i in range(len(row) - 1):
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] *= 2
            row.pop(i+1)
            row.append(0)
    return row

def move_up(grid):
    initial = deepcopy(grid)
    for col in range(len(grid[0])):
        temp_column = [grid[row][col] for row in range(len(grid))]
        temp_column = [tile for tile in temp_column if tile != 0]
        temp_column = merge_tiles(temp_column)
        temp_column += [0] * (len(grid) - len(temp_column))
        for row in range(len(grid)):
            grid[row][col] = temp_column[row]
    if initial != grid:
        grid = random_generate_2(grid)
    return grid

def move_down(grid):
    initial = deepcopy(grid)
    for col in range(len(grid[0])):
        temp_column = [grid[row][col] for row in range(len(grid) - 1, -1, -1)]
        temp_column = [tile for tile in temp_column if tile != 0]
        temp_column = merge_tiles(temp_column)
        temp_column += [0] * (len(grid) - len(temp_column))
        for row in range(len(grid)):
            grid[len(grid) - 1 - row][col] = temp_column[row]
    if initial != grid:
        grid = random_generate_2(grid)
    return grid

def move_left(grid):
    initial = deepcopy(grid)
    for row in range(len(grid)):
        temp_row = [tile for tile in grid[row] if tile != 0]
        temp_row = merge_tiles(temp_row)
        temp_row += [0] * (len(grid) - len(temp_row))
        grid[row] = temp_row
    if initial != grid:
        grid = random_generate_2(grid)
    return grid

def move_right(grid):
    initial = deepcopy(grid)
    for row in range(len(grid)):
        temp_row = [tile for tile in grid[row][::-1] if tile != 0]
        temp_row = merge_tiles(temp_row)
        temp_row += [0] * (len(grid) - len(temp_row))
        grid[row] = temp_row[::-1]
    if initial != grid:
        grid = random_generate_2(grid)
    return grid


def generate_next_states(current_state):
    next_states = []
    g_curr = deepcopy(current_state[2])
    temp_grid = deepcopy(current_state[0])
    moved_grid = move_left(temp_grid)
    if moved_grid != current_state[0]:
        m = 0
        k = 0

        for row in moved_grid:
            for j in row:
                if j > m:
                    k = m
                    m = j
                elif j > k:
                    k = j

        h = 0
        for row in moved_grid:
            h = h + sum(row)

        log_m = math.log(m, 2) if m > 0 else 0
        log_k = math.log(k, 2) if k > 0 else 0

        next_state = [moved_grid, current_state[1] + [current_state[0]], g_curr + 1, current_state[2] + h, log_m, log_k]
        next_states.append(next_state)

    temp_grid = deepcopy(current_state[0])
    moved_grid = move_right(temp_grid)

    if moved_grid != current_state[0]:
        m = 0
        k = 0

        for row in moved_grid:
            for j in row:
                if j > m:
                    k = m
                    m = j
                elif j > k:
                    k = j

        h = 0
        for row in moved_grid:
            h = h + sum(row)

        log_m = math.log(m, 2) if m > 0 else 0
        log_k = math.log(k, 2) if k > 0 else 0

        next_state = [moved_grid, current_state[1] + [current_state[0]], g_curr + 1, current_state[2] + h, log_m, log_k]
        next_states.append(next_state)

    temp_grid = deepcopy(current_state[0])
    moved_grid = move_up(temp_grid)

    if moved_grid != current_state[0]:
        m = 0
        k = 0

        for row in moved_grid:
            for j in row:
                if j > m:
                    k = m
                    m = j
                elif j > k:
                    k = j

        h = 0
        for row in moved_grid:
            h = h + sum(row)

        log_m = math.log(m, 2) if m > 0 else 0
        log_k = math.log(k, 2) if k > 0 else 0

        next_state = [moved_grid, current_state[1] + [current_state[0]], g_curr + 1, current_state[2] + h, log_m, log_k]
        next_states.append(next_state)

    temp_grid = deepcopy(current_state[0])
    moved_grid = move_down(temp_grid)
    if moved_grid != current_state[0]:
        m = 0
        k = 0

        for row in moved_grid:
            for j in row:
                if j > m:
                    k = m
                    m = j
                elif j > k:
                    k = j

        h = 0
        for row in moved_grid:
            h = h + sum(row)

        log_m = math.log(m, 2) if m > 0 else 0
        log_k = math.log(k, 2) if k > 0 else 0

        next_state = [moved_grid, current_state[1] + [current_state[0]], g_curr + 1, current_state[2] + h, log_m, log_k]
        next_states.append(next_state)
    return next_states

grid = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

random_start_pieces = []

rr = random.randint(2, 3)

for i in range(rr):
    x = random.randint(0, 3)
    y = random.randint(0, 3)
    tup = (x, y)
    if tup not in random_start_pieces:
        random_start_pieces.append(tup)
        grid[x][y] = 2

print("Original Grid:")
for row in grid:
    print(row)
print("+=================+")

start_state = [grid, [], 0, 0, 0, 0]


def a_star_algo(start):
    opened = []
    opened.append(start)
    closed = []
    cnt =[]
    values=[]
    count = 1
    while opened:
        opened = sorted(opened, key=lambda x: x[4:])
        current_state = opened.pop()
        closed.append(current_state)
        # if current_state[4] == current_state[5]:
        #     print("===========================")
        #     print(current_state[0], count)
        #     print("===========================")
        final_point = 2048
        cnt.append(count)
        values.append(2**current_state[4])
        for col in range(len(current_state[0])):
            for row in range(len(current_state[0][0])):
                if current_state[0][row][col] == final_point:
                    current_state[1] += [current_state[0]]
                    return current_state, count , cnt, values
        count += 1

        # print(count)
        # current_grid_condition = current_state[0]
        # print(current_grid_condition)
        # print("+++++++++++++++++++++++++++++++")
        if count >= 2000:
            return current_state

        next_states = generate_next_states(current_state)
        if count == 457:
            random_next_states = next_states
        for state in next_states:
            opened.append(state)

    return None, count

# DFS Algorithm
def dfs(start):
    opened = []
    opened.append(start)
    closed = []
    cnt =[]
    values=[]
    count = 1
    while opened:
        current_state = opened.pop()
        closed.append(current_state)
        # if current_state[4] == current_state[5]:
        #     print("===========================")
        #     print(current_state[0], count)
        #     print("===========================")
        final_point = 2048
        cnt.append(count)
        values.append(2**current_state[4])
        for col in range(len(current_state[0])):
            for row in range(len(current_state[0][0])):
                if current_state[0][row][col] == final_point:
                    current_state[1] += [current_state[0]]
                    return current_state, count , cnt, values
        count += 1

        # print(count)
        # current_grid_condition = current_state[0]
        # print(current_grid_condition)
        # print("+++++++++++++++++++++++++++++++")
        if count >= 2000:
            return current_state

        next_states = generate_next_states(current_state)
        for state in next_states:
            opened.append(state)

    return None, count ,0 ,0

# BFS Algorithm
def bfs(start):
    opened = []
    opened.append(start)
    closed = []
    cnt =[]
    values=[]
    count = 1
    while opened:
        current_state = opened.pop(0)
        closed.append(current_state)
        # if current_state[4] == current_state[5]:
        #     print("===========================")
        #     print(current_state[0], count)
        #     print("===========================")
        final_point = 2048
        cnt.append(count)
        values.append(2**current_state[4])
        for col in range(len(current_state[0])):
            for row in range(len(current_state[0][0])):
                if current_state[0][row][col] == final_point:
                    current_state[1] += [current_state[0]]
                    return current_state, count , cnt, values
        count += 1

        # print(count)
        # current_grid_condition = current_state[0]
        # print(current_grid_condition)
        # print("+++++++++++++++++++++++++++++++")
        if count >= 2000:
            return current_state,count,cnt, values

        next_states = generate_next_states(current_state)
        for state in next_states:
            opened.append(state)

    return None, count ,0 ,0


final, final_count, a, b = a_star_algo(start_state)



print("\n\nA star algorithm\n\n")

print(final[0])
for x in final[1]:
    for y in x:
        print(y)
    print("\n")
print(final[2])
print(final[3])
print(final[4])
print(final[5])

print(f"Final count : {final_count}")

plt.title("A* Algorithm")
plt.xlabel("Iterations")
plt.ylabel("Value")

plt.plot(a,b)
plt.show()


final1, final_count1, a1, b1 = dfs(start_state)

print("\n\nDFS Algorithm\n\n")

print(final1[0])
for x in final1[1]:
    for y in x:
        print(y)
    print("\n")
print(final1[2])
print(final1[3])
print(final1[4])
print(final1[5])

print(f"Final count : {final_count1}")

plt.title("Dfs Algorithm")
plt.xlabel("Iterations")
plt.ylabel("Value")

plt.plot(a1,b1)
plt.show()

final2, final_count2, a2, b2 = bfs(start_state)

print("\n\nBFS Algorithm\n\n")

print(final2[0])
for x in final2[1]:
    for y in x:
        print(y)
    print("\n")
print(final2[2])
print(final2[3])
print(final2[4])
print(final2[5])

print(f"Final count : {final_count2}")

plt.title("Bfs Algorithm")
plt.xlabel("Iterations")
plt.ylabel("Value")

plt.plot(a2,b2)
plt.show()




# for i in start_state[0]:
#     print(i)
# next_states=generate_next_states(start_state)
# for i in range(len(next_states)):
#     print("+======================+")
#     for j in next_states[i][0]:
#         print(j)
