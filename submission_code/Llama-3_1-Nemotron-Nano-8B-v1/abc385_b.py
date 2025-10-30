H, W, X, Y = map(int, input().split())
x = X - 1
y = Y - 1
grid = [input().strip() for _ in range(H)]
T = input().strip()

visited = set()

for move in T:
    dx, dy = 0, 0
    if move == 'U':
        dx = -1
    elif move == 'D':
        dx = 1
    elif move == 'L':
        dy = -1
    elif move == 'R':
        dy = 1
    next_x = x + dx
    next_y = y + dy
    if 0 <= next_x < H and 0 <= next_y < W and grid[next_x][next_y] != '#':
        x, y = next_x, next_y
    if grid[x][y] == '@':
        visited.add((x, y))

print(x + 1, y + 1, len(visited))