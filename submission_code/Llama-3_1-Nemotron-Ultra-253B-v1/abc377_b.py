# Read the grid from input
grid = [input().strip() for _ in range(8)]

# Determine which rows are safe (no '#')
row_safe = ['#' not in row for row in grid]

# Determine which columns are safe (no '#' in the column)
col_safe = [all(grid[i][j] != '#' for i in range(8)) for j in range(8)]

# Count the valid positions
count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and row_safe[i] and col_safe[j]:
            count += 1

# Output the result
print(count)