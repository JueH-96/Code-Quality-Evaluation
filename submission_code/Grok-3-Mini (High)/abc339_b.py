import sys

# Read input
data = sys.stdin.read().split()
H, W, N = map(int, data)

# Define direction deltas: up, right, down, left
delta_row = [-1, 0, 1, 0]
delta_col = [0, 1, 0, -1]

# Initialize grid with all white (0)
grid = [[0 for _ in range(W)] for _ in range(H)]

# Initial position and direction
pos_row = 0
pos_col = 0
dir_index = 0  # 0: up, 1: right, 2: down, 3: left

# Perform N operations
for _ in range(N):
    if grid[pos_row][pos_col] == 0:  # White
        grid[pos_row][pos_col] = 1  # Paint black
        dir_index = (dir_index + 1) % 4  # Rotate clockwise
    else:  # Black
        grid[pos_row][pos_col] = 0  # Paint white
        dir_index = (dir_index - 1) % 4  # Rotate counterclockwise
    
    # Move forward in the current direction
    dr = delta_row[dir_index]
    dc = delta_col[dir_index]
    pos_row = (pos_row + dr) % H
    pos_col = (pos_col + dc) % W

# Output the grid
for row in grid:
    print(''.join(['.' if cell == 0 else '#' for cell in row]))