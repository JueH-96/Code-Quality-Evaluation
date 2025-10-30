H, W = map(int, input().split())
Si, Sj = map(int, input().split())
grid = [list(input()) for _ in range(H)]
X = input()

# Convert to 0-based indexing
curr_i = Si - 1
curr_j = Sj - 1

for move in X:
    if move == 'L':
        # Check if we can move left
        if curr_j > 0 and grid[curr_i][curr_j-1] == '.':
            curr_j -= 1
    elif move == 'R':
        # Check if we can move right
        if curr_j < W-1 and grid[curr_i][curr_j+1] == '.':
            curr_j += 1
    elif move == 'U':
        # Check if we can move up
        if curr_i > 0 and grid[curr_i-1][curr_j] == '.':
            curr_i -= 1
    elif move == 'D':
        # Check if we can move down
        if curr_i < H-1 and grid[curr_i+1][curr_j] == '.':
            curr_i += 1

# Convert back to 1-based indexing for output
print(curr_i + 1, curr_j + 1)