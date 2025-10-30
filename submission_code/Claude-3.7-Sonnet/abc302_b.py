def find_snuke(grid, h, w):
    directions = [
        (0, 1),   # Right
        (1, 0),   # Down
        (0, -1),  # Left
        (-1, 0),  # Up
        (1, 1),   # Down-Right
        (-1, -1), # Up-Left
        (1, -1),  # Down-Left
        (-1, 1)   # Up-Right
    ]
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 's':
                for dr, dc in directions:
                    # Check if the pattern extends in this direction
                    if (0 <= i+dr < h and 0 <= j+dc < w and grid[i+dr][j+dc] == 'n' and
                        0 <= i+2*dr < h and 0 <= j+2*dc < w and grid[i+2*dr][j+2*dc] == 'u' and
                        0 <= i+3*dr < h and 0 <= j+3*dc < w and grid[i+3*dr][j+3*dc] == 'k' and
                        0 <= i+4*dr < h and 0 <= j+4*dc < w and grid[i+4*dr][j+4*dc] == 'e'):
                        return [(i+k*dr+1, j+k*dc+1) for k in range(5)]  # +1 to convert to 1-indexed
    
    return None

# Read input
h, w = map(int, input().split())
grid = [input() for _ in range(h)]

# Find the pattern
positions = find_snuke(grid, h, w)

# Print the result
for position in positions:
    print(position[0], position[1])