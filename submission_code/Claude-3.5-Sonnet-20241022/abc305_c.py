H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]

# Find the rectangle boundaries
min_row = H
max_row = 0
min_col = W 
max_col = 0

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            min_row = min(min_row, i)
            max_row = max(max_row, i)
            min_col = min(min_col, j)
            max_col = max(max_col, j)

# Check each position inside the rectangle
for i in range(min_row, max_row + 1):
    for j in range(min_col, max_col + 1):
        if grid[i][j] == '.':
            # This must be where the cookie was eaten
            print(i + 1, j + 1)
            exit()