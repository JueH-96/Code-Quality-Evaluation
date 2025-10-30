# Read the grid
grid = [input().strip() for _ in range(8)]

# Precompute which rows and columns have at least one '#'
row_has_piece = [False] * 8
for i in range(8):
    if '#' in grid[i]:
        row_has_piece[i] = True

col_has_piece = [False] * 8
for j in range(8):
    for i in range(8):
        if grid[i][j] == '#':
            col_has_piece[j] = True
            break  # No need to check further once found

# Count valid positions
count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and not row_has_piece[i] and not col_has_piece[j]:
            count += 1

print(count)