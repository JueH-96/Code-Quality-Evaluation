# Read grid dimensions
H, W = map(int, input().split())

# Read starting position and convert to 0-based indexing
S_i, S_j = map(int, input().split())
current_i = S_i - 1
current_j = S_j - 1

# Read the grid
grid = [input().strip() for _ in range(H)]

# Read movement sequence
X = input().strip()

# Define movement directions
move_direction = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

# Process each movement in the sequence
for move in X:
    di, dj = move_direction.get(move, (0, 0))
    new_i = current_i + di
    new_j = current_j + dj
    # Check if the new position is within bounds and the cell is empty
    if 0 <= new_i < H and 0 <= new_j < W and grid[new_i][new_j] == '.':
        current_i, current_j = new_i, new_j

# Convert back to 1-based indexing and print the final position
print(current_i + 1, current_j + 1)