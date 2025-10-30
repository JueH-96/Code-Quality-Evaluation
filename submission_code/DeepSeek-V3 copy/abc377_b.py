# Read the grid
grid = [input().strip() for _ in range(8)]

# Initialize sets to keep track of rows and columns that are under attack
attacked_rows = set()
attacked_cols = set()

# Iterate through the grid to find all rows and columns that are under attack
for i in range(8):
    for j in range(8):
        if grid[i][j] == '#':
            attacked_rows.add(i)
            attacked_cols.add(j)

# Count the number of safe squares
safe_count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and i not in attacked_rows and j not in attacked_cols:
            safe_count += 1

print(safe_count)