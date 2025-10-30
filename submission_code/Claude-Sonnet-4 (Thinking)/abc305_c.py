H, W = map(int, input().split())
grid = []
for i in range(H):
    grid.append(input().strip())

# Find all cookie positions
cookies = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            cookies.append((i, j))

# Find bounding rectangle
min_row = min(pos[0] for pos in cookies)
max_row = max(pos[0] for pos in cookies)
min_col = min(pos[1] for pos in cookies)
max_col = max(pos[1] for pos in cookies)

# Check each position in the bounding rectangle
found = False
for i in range(min_row, max_row + 1):
    for j in range(min_col, max_col + 1):
        if grid[i][j] == '.':
            print(i + 1, j + 1)  # Convert to 1-indexed
            found = True
            break
    if found:
        break