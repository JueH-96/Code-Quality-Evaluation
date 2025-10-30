# YOUR CODE HERE
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
min_row = min(cell[0] for cell in black_cells)
max_row = max(cell[0] for cell in black_cells)
min_col = min(cell[1] for cell in black_cells)
max_col = max(cell[1] for cell in black_cells)

# Check if it's possible to form a rectangle
possible = True

# Check all cells in the grid
for i in range(H):
    for j in range(W):
        if min_row <= i <= max_row and min_col <= j <= max_col:
            # Cell should be inside the rectangle (black)
            if grid[i][j] == '.':
                # Already white, but needs to be black - impossible
                possible = False
                break
        else:
            # Cell should be outside the rectangle (white)
            if grid[i][j] == '#':
                # Already black, but needs to be white - impossible
                possible = False
                break
    if not possible:
        break

if possible:
    print("Yes")
else:
    print("No")