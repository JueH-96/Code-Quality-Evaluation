# Read input
H, W = map(int, input().split())
Si, Sj = map(int, input().split())

# Read grid
grid = []
for i in range(H):
    grid.append(input().strip())

X = input().strip()

# Current position (convert to 0-indexed)
curr_i = Si - 1
curr_j = Sj - 1

# Process each action
for action in X:
    if action == 'L':
        # Move left
        new_j = curr_j - 1
        if new_j >= 0 and grid[curr_i][new_j] == '.':
            curr_j = new_j
    elif action == 'R':
        # Move right
        new_j = curr_j + 1
        if new_j < W and grid[curr_i][new_j] == '.':
            curr_j = new_j
    elif action == 'U':
        # Move up
        new_i = curr_i - 1
        if new_i >= 0 and grid[new_i][curr_j] == '.':
            curr_i = new_i
    elif action == 'D':
        # Move down
        new_i = curr_i + 1
        if new_i < H and grid[new_i][curr_j] == '.':
            curr_i = new_i

# Convert back to 1-indexed and print
print(curr_i + 1, curr_j + 1)