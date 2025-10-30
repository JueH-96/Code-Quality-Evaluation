import sys

def is_tak_code(grid, i, j):
    if i + 8 >= len(grid) or j + 8 >= len(grid[0]):
        return False

    # Check top-left and bottom-right 3x3 regions
    for x in range(i, i + 3):
        for y in range(j, j + 3):
            if grid[x][y] != '#':
                return False
    for x in range(i + 6, i + 9):
        for y in range(j + 6, j + 9):
            if grid[x][y] != '#':
                return False

    # Check adjacent cells
    for x in range(i - 1, i + 10):
        for y in [j - 1, j + 8]:
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != '.':
                return False
    for x in [i - 1, i + 9]:
        for y in range(j, j + 9):
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != '.':
                return False

    return True

def solve():
    N, M = map(int, input().split())
    grid = [input() for _ in range(N)]

    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(grid, i, j):
                print(i + 1, j + 1)

solve()