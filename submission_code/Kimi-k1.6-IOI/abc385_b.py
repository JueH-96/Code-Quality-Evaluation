H, W, X, Y = map(int, input().split())
grid = [input().strip() for _ in range(H)]
T = input().strip()

current_x = X - 1
current_y = Y - 1
visited = set()

directions = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

for c in T:
    dx, dy = directions[c]
    nx = current_x + dx
    ny = current_y + dy
    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
        current_x, current_y = nx, ny
    if grid[current_x][current_y] == '@':
        visited.add((current_x, current_y))

print(current_x + 1, current_y + 1, len(visited))