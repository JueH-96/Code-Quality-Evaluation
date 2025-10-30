import sys

# Read H and W
H, W = map(int, sys.stdin.readline().split())

# Read the grid
grid = []
for _ in range(H):
    grid.append(sys.stdin.readline().strip())

# Find min/max row and column indices of '#'
min_r, max_r = H, -1
min_c, max_c = W, -1

# Problem guarantees at least one '#'.
# We don't need to handle the case where no '#' is found explicitly
# because min_r, max_r, min_c, max_c will be updated to valid indices.
# The ranges min_r to max_r and min_c to max_c will be valid and non-empty.

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            min_r = min(min_r, i)
            max_r = max(max_r, i)
            min_c = min(min_c, j)
            max_c = max(max_c, j)

# Check if any '.' is inside the minimal bounding box of '#'
# The bounding box is inclusive: rows from min_r to max_r, columns from min_c to max_c (0-based).
for i in range(min_r, max_r + 1):
    for j in range(min_c, max_c + 1):
        if grid[i][j] == '.':
            print("No")
            sys.exit() # Found a contradiction, impossible

# If the loop completes, it means no '.' was found inside the required black rectangle area.
# Since all '#' are guaranteed to be within this bounding box, and there are no '.'s
# conflicting within this box, we can paint all '?' within this box '#' and
# all '?' outside this box '.' to form a valid black rectangle.
print("Yes")