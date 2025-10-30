# Read input
grid = []
for _ in range(8):
    grid.append(input().strip())

# Identify rows and columns with pieces
rows_with_pieces = set()
cols_with_pieces = set()

for i in range(8):
    for j in range(8):
        if grid[i][j] == '#':
            rows_with_pieces.add(i)
            cols_with_pieces.add(j)

# Calculate safe squares
safe_rows = set(range(8)) - rows_with_pieces
safe_cols = set(range(8)) - cols_with_pieces

# The safe squares are those at the intersection of safe rows and safe columns
safe_count = len(safe_rows) * len(safe_cols)

print(safe_count)