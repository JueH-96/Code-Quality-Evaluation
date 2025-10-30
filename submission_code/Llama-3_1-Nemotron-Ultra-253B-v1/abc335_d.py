n = int(input())
c = (n + 1) // 2
spiral = []

for layer in range((n + 1) // 2):
    start = 1 + layer
    end = n - layer
    # Top row: left to right
    for col in range(start, end + 1):
        spiral.append((start, col))
    # Right column: top+1 to end
    for row in range(start + 1, end + 1):
        spiral.append((row, end))
    if start < end:
        # Bottom row: right-1 to left
        for col in range(end - 1, start - 1, -1):
            spiral.append((end, col))
        # Left column: bottom-1 to top+1
        for row in range(end - 1, start, -1):
            spiral.append((row, start))

# Remove the center cell (last element)
if spiral:
    spiral.pop()

# Create the grid
grid = [['T' for _ in range(n)] for _ in range(n)]
for i, (x, y) in enumerate(spiral, 1):
    grid[x - 1][y - 1] = i

# Print the result
for row in grid:
    print(' '.join(map(str, row)))