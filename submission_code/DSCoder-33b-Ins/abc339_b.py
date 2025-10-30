# YOUR CODE HERE
H, W, N = map(int, input().split())
grid = [['.' for _ in range(W)] for _ in range(H)]

x, y, dx, dy = 0, 0, 0, 1
for _ in range(N):
    grid[x][y] = '#' if grid[x][y] == '.' else '.'
    if x + dx < 0 or x + dx >= H or y + dy < 0 or y + dy >= W or grid[x+dx][y+dy] == '#':
        dx, dy = dy, -dx
    x, y = x + dx, y + dy

for row in grid:
    print(''.join(row))