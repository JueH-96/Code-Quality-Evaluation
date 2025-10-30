# Read the grid
grid = [input().strip() for _ in range(8)]

# Find safe rows (rows with no '#')
safe_rows = [i for i in range(8) if '#' not in grid[i]]

# Find safe columns (columns with no '#')
safe_cols = [j for j in range(8) if all(grid[i][j] != '#' for i in range(8))]

# Count the number of safe positions
count = 0
for i in safe_rows:
    for j in safe_cols:
        if grid[i][j] == '.':
            count += 1

print(count)