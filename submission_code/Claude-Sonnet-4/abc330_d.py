n = int(input())
grid = []
for _ in range(n):
    grid.append(input().strip())

count = 0

# Find all cells with 'o'
o_cells = []
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            o_cells.append((i, j))

# For each cell with 'o', consider it as the corner of an L-shape
for i, j in o_cells:
    # Count cells with 'o' in the same row (excluding current cell)
    same_row = []
    for k in range(n):
        if k != j and grid[i][k] == 'o':
            same_row.append(k)
    
    # Count cells with 'o' in the same column (excluding current cell)
    same_col = []
    for m in range(n):
        if m != i and grid[m][j] == 'o':
            same_col.append(m)
    
    # Each combination of one cell from same_row and one from same_col
    # forms a valid triple with the current cell
    count += len(same_row) * len(same_col)

print(count)