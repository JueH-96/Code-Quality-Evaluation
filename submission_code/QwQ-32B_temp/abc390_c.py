H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

min_row = H  # Initialize to a value larger than any possible row
max_row = -1
min_col = W  # Initialize to a value larger than any possible column
max_col = -1

# Find the boundaries of the existing black cells
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            if i < min_row:
                min_row = i
            if i > max_row:
                max_row = i
            if j < min_col:
                min_col = j
            if j > max_col:
                max_col = j

# Check if any '.' exists within the required rectangle
possible = True
for i in range(min_row, max_row + 1):
    for j in range(min_col, max_col + 1):
        if grid[i][j] == '.':
            possible = False
            break
    if not possible:
        break

print("Yes" if possible else "No")