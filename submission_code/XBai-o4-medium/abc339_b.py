H, W, N = map(int, input().split())

grid = [[0 for _ in range(W)] for _ in range(H)]
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
i, j = 0, 0
d = 0  # initially facing up

for _ in range(N):
    current_color = grid[i][j]
    # Flip the color
    grid[i][j] = 1 - current_color
    # Rotate direction
    if current_color == 0:
        d = (d + 1) % 4
    else:
        d = (d - 1) % 4
    # Move
    di, dj = directions[d]
    i = (i + di) % H
    j = (j + dj) % W

for row in grid:
    print(''.join(['#' if cell else '.' for cell in row]))