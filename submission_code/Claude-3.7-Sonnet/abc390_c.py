# Read input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Find the boundary of the potential rectangle
min_row, max_row, min_col, max_col = H, -1, W, -1

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            min_row = min(min_row, i)
            max_row = max(max_row, i)
            min_col = min(min_col, j)
            max_col = max(max_col, j)

possible = True

# Check if there are any white cells inside the potential rectangle
# or any black cells outside it
for i in range(H):
    for j in range(W):
        inside_rectangle = min_row <= i <= max_row and min_col <= j <= max_col
        
        if grid[i][j] == '#' and not inside_rectangle:
            possible = False
            break
        if grid[i][j] == '.' and inside_rectangle:
            possible = False
            break
    if not possible:
        break

print("Yes" if possible else "No")