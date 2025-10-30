# Read input values
H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
grid = [input().strip() for _ in range(H)]
X = input().strip()

# Adjust to 0-based index
current_i = S_i - 1
current_j = S_j - 1

# Define movement directions
directions = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

# Iterate through each character in X
for move in X:
    di, dj = directions[move]
    new_i = current_i + di
    new_j = current_j + dj
    # Check if new position is within grid and is empty
    if 0 <= new_i < H and 0 <= new_j < W and grid[new_i][new_j] == '.':
        current_i, current_j = new_i, new_j

# Convert back to 1-based index
print(current_i + 1, current_j + 1)