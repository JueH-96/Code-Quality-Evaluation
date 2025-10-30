H, W, N = map(int, input().split())

# Initialize the grid with all white cells
grid = [['.' for _ in range(W)] for _ in range(H)]
i, j = 0, 0  # Starting position is (0,0) in 0-based index
direction = 0  # 0: up, 1: right, 2: down, 3: left

for _ in range(N):
    current_color = grid[i][j]
    
    if current_color == '.':
        grid[i][j] = '#'
        direction = (direction + 1) % 4
    else:
        grid[i][j] = '.'
        direction = (direction - 1) % 4
    
    # Update position based on the new direction
    if direction == 0:
        i = (i - 1) % H
    elif direction == 1:
        j = (j + 1) % W
    elif direction == 2:
        i = (i + 1) % H
    else:
        j = (j - 1) % W

# Print the resulting grid
for row in grid:
    print(''.join(row))