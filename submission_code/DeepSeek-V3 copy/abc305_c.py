# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Determine the boundaries of the rectangle
a = b = c = d = None

# Find the top row (a)
for i in range(H):
    if '#' in grid[i]:
        a = i
        break

# Find the bottom row (b)
for i in range(H-1, -1, -1):
    if '#' in grid[i]:
        b = i
        break

# Find the left column (c)
for j in range(W):
    if any(grid[i][j] == '#' for i in range(H)):
        c = j
        break

# Find the right column (d)
for j in range(W-1, -1, -1):
    if any(grid[i][j] == '#' for i in range(H)):
        d = j
        break

# Now, find the missing cookie
for i in range(a, b+1):
    for j in range(c, d+1):
        if grid[i][j] != '#':
            print(i+1, j+1)
            exit()