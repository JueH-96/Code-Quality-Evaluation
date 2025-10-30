# YOUR CODE HERE

import sys

# Read the inputs
H, W = map(int, sys.stdin.readline().split())
S_i, S_j = map(int, sys.stdin.readline().split())
grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
actions = sys.stdin.readline().strip()

# Define the directions
directions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

# Perform the actions
for action in actions:
    di, dj = directions[action]
    ni, nj = S_i + di, S_j + dj
    if 1 <= ni <= H and 1 <= nj <= W and grid[ni - 1][nj - 1] == '.':
        S_i, S_j = ni, nj

# Print the final position
print(S_i, S_j)