h, w = map(int, input().split())
grid = [input().strip() for _ in range(h)]

# Find all rows that contain at least one '#'
rows_with_hash = [i for i in range(h) if '#' in grid[i]]
a = min(rows_with_hash)
b = max(rows_with_hash)

# Find all columns that contain at least one '#'
columns_with_hash = []
for i in range(h):
    for j in range(w):
        if grid[i][j] == '#':
            columns_with_hash.append(j)
c = min(columns_with_hash)
d = max(columns_with_hash)

# Search for the missing cookie within the determined rectangle
for i in range(a, b + 1):
    for j in range(c, d + 1):
        if grid[i][j] == '.':
            print(i + 1, j + 1)
            exit()