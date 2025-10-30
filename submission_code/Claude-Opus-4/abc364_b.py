# YOUR CODE HERE
# Read input
H, W = map(int, input().split())
S_i, S_j = map(int, input().split())

# Read the grid
grid = []
for _ in range(H):
    grid.append(input().strip())

# Read movement string
X = input().strip()

# Current position (1-indexed to 0-indexed)
row = S_i - 1
col = S_j - 1

# Process each movement
for move in X:
    if move == 'L':
        # Try to move left
        new_col = col - 1
        if new_col >= 0 and grid[row][new_col] == '.':
            col = new_col
    elif move == 'R':
        # Try to move right
        new_col = col + 1
        if new_col < W and grid[row][new_col] == '.':
            col = new_col
    elif move == 'U':
        # Try to move up
        new_row = row - 1
        if new_row >= 0 and grid[new_row][col] == '.':
            row = new_row
    elif move == 'D':
        # Try to move down
        new_row = row + 1
        if new_row < H and grid[new_row][col] == '.':
            row = new_row

# Output final position (0-indexed to 1-indexed)
print(row + 1, col + 1)