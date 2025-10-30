import sys
data = sys.stdin.read().split()
H = int(data[0])
W = int(data[1])
grid = data[2:2+H]

# Find the bounding box of all '#' cells
min_row = H
max_row = -1
min_col = W
max_col = -1

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            min_row = min(min_row, i)
            max_row = max(max_row, i)
            min_col = min(min_col, j)
            max_col = max(max_col, j)

# Check if there is any '.' in the bounding box
if any(grid[i][j] == '.' for i in range(min_row, max_row + 1) for j in range(min_col, max_col + 1)):
    print("No")
else:
    print("Yes")