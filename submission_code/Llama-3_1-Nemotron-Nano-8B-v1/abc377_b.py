# Read the grid
grid = [input().strip() for _ in range(8)]

count = 0

for i in range(8):
    for j in range(8):
        if grid[i][j] != '.':
            continue
        # Check if the entire row is empty
        row_empty = all(c == '.' for c in grid[i])
        # Check if the entire column is empty
        col_empty = all(grid[k][j] == '.' for k in range(8))
        if row_empty and col_empty:
            count += 1

print(count)