# Read the grid
grid = [input().strip() for _ in range(8)]

# Determine safe rows (1-based)
safe_rows = set()
for i in range(8):
    if '#' not in grid[i]:
        safe_rows.add(i + 1)

# Determine safe columns (1-based)
safe_cols = set()
for j in range(8):
    has_piece = False
    for i in range(8):
        if grid[i][j] == '#':
            has_piece = True
            break
    if not has_piece:
        safe_cols.add(j + 1)

# Count valid squares
count = 0
for i in range(8):
    for j in range(8):
        row = i + 1
        col = j + 1
        if row in safe_rows and col in safe_cols and grid[i][j] == '.':
            count += 1

print(count)