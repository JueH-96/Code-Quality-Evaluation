H, W, N = map(int, input().split())

grid = [list('.' * W) for _ in range(H)]
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
current_i, current_j = 0, 0
direction_index = 0  # Initial direction is up

for _ in range(N):
    original_color = grid[current_i][current_j]
    # Repaint the cell
    grid[current_i][current_j] = '#' if original_color == '.' else '.'
    # Rotate direction
    if original_color == '.':
        direction_index = (direction_index + 1) % 4
    else:
        direction_index = (direction_index - 1) % 4
    # Move to next cell
    di, dj = directions[direction_index]
    current_i = (current_i + di) % H
    current_j = (current_j + dj) % W

for row in grid:
    print(''.join(row))