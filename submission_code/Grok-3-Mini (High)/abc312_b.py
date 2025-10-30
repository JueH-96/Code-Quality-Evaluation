import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
grid = []
for _ in range(N):
    s = data[index]
    grid.append(s)
    index += 1

# Define offsets for black and white cells in the 9x9 subgrid
black_offsets = [(i, j) for i in range(3) for j in range(3)] + [(i, j) for i in range(6, 9) for j in range(6, 9)]
white_offsets = [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3),
                 (5, 5), (5, 6), (5, 7), (5, 8), (6, 5), (7, 5), (8, 5)]

# Iterate over all possible top-left positions for 9x9 subgrid
for r_start in range(N - 8):  # r_start from 0 to N-9 inclusive
    for c_start in range(M - 8):  # c_start from 0 to M-9 inclusive
        # Check all black cells are '#'
        all_black = all(grid[r_start + di][c_start + dj] == '#' for di, dj in black_offsets)
        # Check all white cells are '.'
        all_white = all(grid[r_start + di][c_start + dj] == '.' for di, dj in white_offsets)
        # If both conditions satisfied, print 1-based coordinates
        if all_black and all_white:
            print(f"{r_start + 1} {c_start + 1}")