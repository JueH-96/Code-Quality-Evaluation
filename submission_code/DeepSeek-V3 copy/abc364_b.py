# Read input
H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
grid = [input().strip() for _ in range(H)]
X = input().strip()

# Adjust to 0-based index
current_i = S_i - 1
current_j = S_j - 1

# Process each move
for move in X:
    if move == 'L':
        if current_j > 0 and grid[current_i][current_j - 1] == '.':
            current_j -= 1
    elif move == 'R':
        if current_j < W - 1 and grid[current_i][current_j + 1] == '.':
            current_j += 1
    elif move == 'U':
        if current_i > 0 and grid[current_i - 1][current_j] == '.':
            current_i -= 1
    elif move == 'D':
        if current_i < H - 1 and grid[current_i + 1][current_j] == '.':
            current_i += 1

# Output the final position (1-based)
print(current_i + 1, current_j + 1)