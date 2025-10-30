n = int(input())

# Initialize the grid
grid = [[0 for _ in range(n)] for __ in range(n)]

# Generate the spiral order
spiral = []
top = 0
bottom = n - 1
left = 0
right = n - 1

while top <= bottom and left <= right:
    # Top row, left to right
    for j in range(left, right + 1):
        spiral.append((top, j))
    top += 1

    # Right column, top to bottom
    for i in range(top, bottom + 1):
        spiral.append((i, right))
    right -= 1

    # Bottom row, right to left if applicable
    if top <= bottom:
        for j in range(right, left - 1, -1):
            spiral.append((bottom, j))
        bottom -= 1

    # Left column, bottom to top if applicable
    if left <= right:
        for i in range(bottom, top - 1, -1):
            spiral.append((i, left))
        left += 1

# Determine the center coordinates
center = (n // 2, n // 2)
# Remove the center if it's the last element in the spiral
if spiral and spiral[-1] == center:
    spiral.pop()

# Assign the numbers to the grid
for idx, (i, j) in enumerate(spiral, 1):
    grid[i][j] = str(idx)

# Place 'T' at the center
grid[center[0]][center[1]] = 'T'

# Print the grid
for row in grid:
    print(' '.join(row))