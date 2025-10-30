# YOUR CODE HERE
# Read input
grid = []
for _ in range(8):
    grid.append(input().strip())

# Find rows with no pieces
safe_rows = []
for i in range(8):
    if '#' not in grid[i]:
        safe_rows.append(i)

# Find columns with no pieces
safe_cols = []
for j in range(8):
    has_piece = False
    for i in range(8):
        if grid[i][j] == '#':
            has_piece = True
            break
    if not has_piece:
        safe_cols.append(j)

# Count empty squares that are in safe rows and safe columns
count = 0
for i in safe_rows:
    for j in safe_cols:
        if grid[i][j] == '.':
            count += 1

print(count)