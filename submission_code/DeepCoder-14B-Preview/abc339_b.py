H, W, N = map(int, input().split())

# Initialize grid with all white (False)
grid = [[False for _ in range(W)] for _ in range(H)]

i, j = 1, 1  # Starting position (1-based)
direction = 0  # 0: up, 1: right, 2: down, 3: left

# Define the direction vectors
directions = [ (-1, 0), (0, 1), (1, 0), (0, -1) ]

for _ in range(N):
    current_i = i - 1
    current_j = j - 1

    if grid[current_i][current_j]:
        # Current cell is black, repaint white
        grid[current_i][current_j] = False
        # Rotate counter-clockwise (direction decreases by 1)
        direction = (direction - 1) % 4
    else:
        # Current cell is white, repaint black
        grid[current_i][current_j] = True
        # Rotate clockwise (direction increases by 1)
        direction = (direction + 1) % 4

    # Move in the new direction
    dr, dc = directions[direction]
    i += dr
    j += dc

    # Handle toroidal movement for rows
    if i < 1:
        i += H
    elif i > H:
        i -= H

    # Handle toroidal movement for columns
    if j < 1:
        j += W
    elif j > W:
        j -= W

# Print the grid
for row in grid:
    print(''.join(['#' if cell else '.' for cell in row]))