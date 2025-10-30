H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Find rows that contain at least one '#'
rows_with_hash = [i for i in range(H) if '#' in grid[i]]
a0 = min(rows_with_hash)
b0 = max(rows_with_hash)

# Find columns that contain at least one '#'
cols_with_hash = [j for j in range(W) if any(grid[i][j] == '#' for i in range(H))]
c0 = min(cols_with_hash)
d0 = max(cols_with_hash)

# Search within the determined rectangle for the missing '.'
for i in range(a0, b0 + 1):
    for j in range(c0, d0 + 1):
        if grid[i][j] == '.':
            print(i + 1, j + 1)
            exit()