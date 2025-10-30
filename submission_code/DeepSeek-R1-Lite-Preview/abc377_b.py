# Read the grid
grid = [input().strip() for _ in range(8)]

# Determine which rows have at least one piece
rows_with_piece = [ '#' in row for row in grid ]

# Determine which columns have at least one piece
cols_with_piece = [ False ] * 8
for j in range(8):
    for i in range(8):
        if grid[i][j] == '#':
            cols_with_piece[j] = True
            break

# Count the number of empty squares where both the row and column have no pieces
count = 0
for i in range(8):
    if not rows_with_piece[i]:
        for j in range(8):
            if not cols_with_piece[j] and grid[i][j] == '.':
                count += 1

# Output the result
print(count)