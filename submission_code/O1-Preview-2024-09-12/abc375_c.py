# YOUR CODE HERE
N = int(input())
grid = [list(input().strip()) for _ in range(N)]
result_grid = [[''] * N for _ in range(N)]

for x in range(N):
    for y in range(N):
        layer = min(x, y, N - 1 - x, N - 1 - y)
        size = N - 2 * layer
        rotations = (N // 2 - layer) % 4
        x_rel = x - layer
        y_rel = y - layer
        for _ in range(rotations):
            x_rel, y_rel = y_rel, size - 1 - x_rel
        x_new = layer + x_rel
        y_new = layer + y_rel
        result_grid[x_new][y_new] = grid[x][y]

for row in result_grid:
    print(''.join(row))