# YOUR CODE HERE
# Read input
H, W = map(int, input().split())
S_i, S_j = map(int, input().split())

# Read grid
grid = []
for i in range(H):
    grid.append(input())

# Read movement string
X = input()

# Current position (1-indexed)
row, col = S_i, S_j

# Direction mappings
directions = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

# Process each movement
for move in X:
    dr, dc = directions[move]
    new_row = row + dr
    new_col = col + dc
    
    # Check if new position is valid
    if 1 <= new_row <= H and 1 <= new_col <= W:
        # Convert to 0-indexed for grid access
        if grid[new_row - 1][new_col - 1] == '.':
            row = new_row
            col = new_col

# Print final position
print(row, col)