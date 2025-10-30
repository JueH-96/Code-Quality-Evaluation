H, W, N = map(int, input().split())

# Initialize grid: False represents white, True represents black
grid = [[False for _ in range(W)] for _ in range(H)]

current_i, current_j = 0, 0
current_dir = 0  # 0: up, 1: right, 2: down, 3: left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

for _ in range(N):
    original = grid[current_i][current_j]
    # Flip the color
    grid[current_i][current_j] = not original
    # Update direction
    if original:
        current_dir = (current_dir - 1) % 4
    else:
        current_dir = (current_dir + 1) % 4
    # Move
    di, dj = directions[current_dir]
    current_i = (current_i + di) % H
    current_j = (current_j + dj) % W

# Convert the grid to the required output format
for row in grid:
    print(''.join('#' if cell else '.' for cell in row))