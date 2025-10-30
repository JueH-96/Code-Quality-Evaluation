# YOUR CODE HERE
import sys
import itertools

def check_line(line):
    return line[0] == line[1] and line[0] != line[2]

def check_grid(grid):
    for i in range(3):
        if check_line(grid[i]):
            return False
        if check_line([grid[j][i] for j in range(3)]):
            return False
    if check_line([grid[i][i] for i in range(3)]):
        return False
    if check_line([grid[i][2-i] for i in range(3)]):
        return False
    return True

def solve():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, sys.stdin.readline().split())))
    valid_grids = 0
    total_grids = 0
    for perm in itertools.permutations(range(1, 10)):
        new_grid = [[perm[i*3+j] for j in range(3)] for i in range(3)]
        if check_grid(new_grid):
            valid_grids += 1
        total_grids += 1
    print(valid_grids / total_grids)

solve()