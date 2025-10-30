# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
H = int(data[0])
W = int(data[1])
S_i = int(data[2])
S_j = int(data[3])
grid = data[4:H+4]
X = data[H+4]

# Convert starting position to 0-based index
current_i = S_i - 1
current_j = S_j - 1

# Define movement directions
directions = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

# Process each move in X
for move in X:
    di, dj = directions[move]
    new_i = current_i + di
    new_j = current_j + dj
    
    # Check if the new position is within bounds and is empty
    if 0 <= new_i < H and 0 <= new_j < W and grid[new_i][new_j] == '.':
        current_i = new_i
        current_j = new_j

# Convert final position back to 1-based index
final_i = current_i + 1
final_j = current_j + 1

print(final_i, final_j)