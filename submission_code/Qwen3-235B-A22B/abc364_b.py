# Read input
H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
current_row = S_i - 1
current_col = S_j - 1

# Read grid
grid = [input().strip() for _ in range(H)]

# Read movement string
X = input().strip()

# Process each movement
for c in X:
    new_row, new_col = current_row, current_col
    if c == 'L':
        new_col -= 1
    elif c == 'R':
        new_col += 1
    elif c == 'U':
        new_row -= 1
    elif c == 'D':
        new_row += 1

    # Check boundaries and cell type
    if 0 <= new_row < H and 0 <= new_col < W:
        if grid[new_row][new_col] == '.':
            current_row, current_col = new_row, new_col

# Output the final position (converted to 1-based)
print(current_row + 1, current_col + 1)