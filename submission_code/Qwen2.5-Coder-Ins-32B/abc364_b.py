# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
H = int(input[0])
W = int(input[1])
S_i = int(input[2]) - 1
S_j = int(input[3]) - 1
grid = [input[4 + i] for i in range(H)]
X = input[4 + H]

directions = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

current_i, current_j = S_i, S_j

for move in X:
    di, dj = directions[move]
    new_i, new_j = current_i + di, current_j + dj
    if 0 <= new_i < H and 0 <= new_j < W and grid[new_i][new_j] == '.':
        current_i, current_j = new_i, new_j

print(current_i + 1, current_j + 1)