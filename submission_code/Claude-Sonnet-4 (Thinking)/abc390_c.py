H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find all black cells
black_cells = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            black_cells.append((i, j))

# Find the bounding rectangle of all black cells
top = min(row for row, col in black_cells)
bottom = max(row for row, col in black_cells)
left = min(col for row, col in black_cells)
right = max(col for row, col in black_cells)

# Check if all cells in this rectangle can be black
for i in range(top, bottom + 1):
    for j in range(left, right + 1):
        if grid[i][j] == '.':  # Already painted white
            print("No")
            exit()

print("Yes")