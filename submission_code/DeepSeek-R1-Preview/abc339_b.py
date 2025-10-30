H, W, N = map(int, input().split())

# Initialize grid with all white cells
grid = [['.' for _ in range(W)] for _ in range(H)]

# Starting position (0, 0) and direction (up)
i, j = 0, 0
direction = 0  # 0: up, 1: right, 2: down, 3: left

# Deltas for each direction: up, right, down, left
deltas = [ (-1, 0), (0, 1), (1, 0), (0, -1) ]

for _ in range(N):
    current_color = grid[i][j]
    if current_color == '.':
        # Paint to black, turn clockwise
        grid[i][j] = '#'
        direction = (direction + 1) % 4
    else:
        # Paint to white, turn counter-clockwise
        grid[i][j] = '.'
        direction = (direction - 1) % 4
    
    # Move in the new direction
    di, dj = deltas[direction]
    i = (i + di) % H
    j = (j + dj) % W

# Print the grid
for row in grid:
    print(''.join(row))