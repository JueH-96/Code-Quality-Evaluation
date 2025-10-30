# Read the grid from input
grid = [input().strip() for _ in range(8)]

# Determine valid rows (rows with no existing pieces)
valid_rows = [False] * 8
for i in range(8):
    valid_rows[i] = '#' not in grid[i]

# Determine valid columns (columns with no existing pieces)
valid_cols = [False] * 8
for j in range(8):
    has_hash = False
    for i in range(8):
        if grid[i][j] == '#':
            has_hash = True
            break
    valid_cols[j] = not has_hash

# Count the valid positions
count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and valid_rows[i] and valid_cols[j]:
            count += 1

# Output the result
print(count)