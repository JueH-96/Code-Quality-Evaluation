# YOUR CODE HERE
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find the bounding rectangle of all cookies
min_row, max_row = H, -1
min_col, max_col = W, -1

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            min_row = min(min_row, i)
            max_row = max(max_row, i)
            min_col = min(min_col, j)
            max_col = max(max_col, j)

# Find the missing cookie within the bounding rectangle
for i in range(min_row, max_row + 1):
    for j in range(min_col, max_col + 1):
        if grid[i][j] == '.':
            # Convert to 1-indexed and print
            print(i + 1, j + 1)
            break