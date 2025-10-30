# YOUR CODE HERE
# Read the 8x8 grid
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

# Count empty squares that are safe (no pieces in their row or column)
safe_squares = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and i not in rows_with_pieces and j not in cols_with_pieces:
            safe_squares += 1

print(safe_squares)