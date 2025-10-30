# YOUR CODE HERE
H, W = map(int, input().split())
Si, Sj = map(int, input().split())

# Read the grid
grid = []
for i in range(H):
    grid.append(input().strip())

X = input().strip()

# Current position (convert to 0-indexed for easier array access)
curr_i = Si - 1
curr_j = Sj - 1

# Process each move
for move in X:
    if move == 'L':
        # Move left (decrease column)
        new_j = curr_j - 1
        if new_j >= 0 and grid[curr_i][new_j] == '.':
            curr_j = new_j
    elif move == 'R':
        # Move right (increase column)
        new_j = curr_j + 1
        if new_j < W and grid[curr_i][new_j] == '.':
            curr_j = new_j
    elif move == 'U':
        # Move up (decrease row)
        new_i = curr_i - 1
        if new_i >= 0 and grid[new_i][curr_j] == '.':
            curr_i = new_i
    elif move == 'D':
        # Move down (increase row)
        new_i = curr_i + 1
        if new_i < H and grid[new_i][curr_j] == '.':
            curr_i = new_i

# Convert back to 1-indexed and output
print(curr_i + 1, curr_j + 1)