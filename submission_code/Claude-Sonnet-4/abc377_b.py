# Read the grid
grid = []
for _ in range(8):
    grid.append(input().strip())

# Find which rows and columns have pieces
rows_with_pieces = set()
cols_with_pieces = set()

for i in range(8):
    for j in range(8):
        if grid[i][j] == '#':
            rows_with_pieces.add(i)
            cols_with_pieces.add(j)

# Count safe empty squares
safe_count = 0

for i in range(8):
    for j in range(8):
        # Check if square is empty
        if grid[i][j] == '.':
            # Check if this square is safe (row i and column j have no pieces)
            if i not in rows_with_pieces and j not in cols_with_pieces:
                safe_count += 1

print(safe_count)