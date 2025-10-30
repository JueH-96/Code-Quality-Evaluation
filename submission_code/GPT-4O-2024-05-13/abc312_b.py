# YOUR CODE HERE
def is_tak_code(grid, i, j):
    # Check top-left 3x3 region
    for x in range(3):
        for y in range(3):
            if grid[i + x][j + y] != '#':
                return False
    # Check bottom-right 3x3 region
    for x in range(6, 9):
        for y in range(6, 9):
            if grid[i + x][j + y] != '#':
                return False
    # Check white cells around top-left 3x3 region
    for x in range(4):
        if grid[i + x][j + 3] != '.':
            return False
    for y in range(4):
        if grid[i + 3][j + y] != '.':
            return False
    # Check white cells around bottom-right 3x3 region
    for x in range(5, 9):
        if grid[i + x][j + 5] != '.':
            return False
    for y in range(5, 9):
        if grid[i + 5][j + y] != '.':
            return False
    return True

def find_tak_codes(N, M, grid):
    results = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(grid, i, j):
                results.append((i + 1, j + 1))
    return results

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
grid = data[2:]

results = find_tak_codes(N, M, grid)
for result in results:
    print(result[0], result[1])