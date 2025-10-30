n = int(input())
center = (n - 1) // 2

# Initialize boundaries for spiral traversal
top = 0
bottom = n - 1
left = 0
right = n - 1
cells = []

# Generate the spiral order traversal of all cells
while top <= bottom and left <= right:
    # Traverse top row from left to right
    for col in range(left, right + 1):
        cells.append((top, col))
    top += 1
    
    # Traverse right column from top to bottom
    for row in range(top, bottom + 1):
        cells.append((row, right))
    right -= 1
    
    # Traverse bottom row from right to left if applicable
    if top <= bottom:
        for col in range(right, left - 1, -1):
            cells.append((bottom, col))
        bottom -= 1
    
    # Traverse left column from bottom to top if applicable
    if left <= right:
        for row in range(bottom, top - 1, -1):
            cells.append((row, left))
        left += 1

# Remove the last element (center cell) to exclude it
cells.pop()

# Initialize the grid
grid = [['0'] * n for _ in range(n)]

# Fill the grid with dragon parts in spiral order
for idx, (i, j) in enumerate(cells):
    grid[i][j] = str(idx + 1)

# Place Takahashi at the center
grid[center][center] = 'T'

# Output the result
for row in grid:
    print(' '.join(row))