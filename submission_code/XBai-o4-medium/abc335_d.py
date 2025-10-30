n = int(input())
order = []
top = 0
bottom = n - 1
left = 0
right = n - 1

while top <= bottom and left <= right:
    # Top row: left to right
    for i in range(left, right + 1):
        order.append((top, i))
    top += 1
    
    # Right column: top to bottom
    for i in range(top, bottom + 1):
        order.append((i, right))
    right -= 1
    
    if top <= bottom:
        # Bottom row: right to left
        for i in range(right, left - 1, -1):
            order.append((bottom, i))
        bottom -= 1
    
    if left <= right:
        # Left column: bottom to top
        for i in range(bottom, top - 1, -1):
            order.append((i, left))
        left += 1

# Remove the last element (the center cell)
order.pop()

# Initialize grid
grid = [['0'] * n for _ in range(n)]

for idx, (r, c) in enumerate(order):
    grid[r][c] = str(idx + 1)

# Set center to 'T'
center = (n - 1) // 2
grid[center][center] = 'T'

# Print the grid
for row in grid:
    print(' '.join(row))