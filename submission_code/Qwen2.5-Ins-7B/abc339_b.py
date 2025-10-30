# YOUR CODE HERE
H, W, N = map(int, input().split())
grid = [['.' for _ in range(W)] for _ in range(H)]
directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
direction = 0
x, y = 0, 0

for _ in range(N):
    if grid[x][y] == '.':
        grid[x][y] = '#'
        direction = (direction + 1) % 4
    else:
        grid[x][y] = '.'
        direction = (direction - 1) % 4
    dx, dy = directions[direction]
    x, y = (x + dx) % H, (y + dy) % W

for row in grid:
    print(''.join(row))