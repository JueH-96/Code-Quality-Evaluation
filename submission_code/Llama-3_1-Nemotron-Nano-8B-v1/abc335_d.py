n = int(input())

path = []
for layer in range((n + 1) // 2):
    start_row = layer
    end_row = n - 1 - layer
    start_col = layer
    end_col = n - 1 - layer

    # Top row: left to right
    for col in range(start_col, end_col + 1):
        path.append((start_row, col))
    
    # Right column: top+1 to bottom
    for row in range(start_row + 1, end_row + 1):
        path.append((row, end_col))
    
    # Bottom row: right-1 to left if applicable
    if start_row < end_row:
        for col in range(end_col - 1, start_col - 1, -1):
            path.append((end_row, col))
    
    # Left column: bottom-1 to top+1 if applicable
    if start_col < end_col:
        for row in range(end_row - 1, start_row, -1):
            path.append((row, start_col))

# Initialize grid with 'T' at the center
grid = [[0] * n for _ in range(n)]
center = (n - 1) // 2
grid[center][center] = 'T'

current = 1
for (r, c) in path:
    grid[r][c] = current
    current += 1

# Print the grid
for row in grid:
    print(' '.join(map(str, row)))