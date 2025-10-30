H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Find the first and last rows containing '#'
a = None
for i in range(H):
    if '#' in grid[i]:
        a = i
        break

b = None
for i in range(H-1, -1, -1):
    if '#' in grid[i]:
        b = i
        break

# Collect the first and last columns for each row in a to b
first_cols = []
last_cols = []
for i in range(a, b + 1):
    row = grid[i]
    first = row.find('#')
    last = row.rfind('#')
    first_cols.append(first)
    last_cols.append(last)

c = min(first_cols)
d = max(last_cols)

# Find the missing cookie within the rectangle
for i in range(a, b + 1):
    for j in range(c, d + 1):
        if grid[i][j] == '.':
            print(i + 1, j + 1)
            exit()