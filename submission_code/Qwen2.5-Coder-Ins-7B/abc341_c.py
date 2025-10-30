# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
N = int(data[2])
T = data[3]
S = data[4:]

# Find the initial position where Takahashi crash-landed
for i in range(H):
    for j in range(W):
        if S[i][j] == '.':
            start_i, start_j = i, j
            break

# Directions for L, R, U, D
directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}

# Function to check if a position is within the grid and not sea
def is_valid(x, y):
    return 0 <= x < H and 0 <= y < W and S[x][y] == '.'

# Simulate all possible initial positions
possible_positions = set()
for i in range(H):
    for j in range(W):
        if S[i][j] == '.':
            # Check if the path from the initial position to the given path is valid
            x, y = i, j
            for move in T:
                dx, dy = directions[move]
                x, y = x + dx, y + dy
                if not is_valid(x, y):
                    break
            else:
                possible_positions.add((x, y))

# Output the number of possible positions
print(len(possible_positions))