# YOUR CODE HERE
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find bounding box of all black cells
min_row, max_row = H, -1
min_col, max_col = W, -1

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            min_row = min(min_row, i)
            max_row = max(max_row, i)
            min_col = min(min_col, j)
            max_col = max(max_col, j)

# Check if all cells in the bounding box are black or ?
possible = True
for i in range(min_row, max_row + 1):
    for j in range(min_col, max_col + 1):
        if grid[i][j] == '.':
            possible = False
            break
    if not possible:
        break

print("Yes" if possible else "No")