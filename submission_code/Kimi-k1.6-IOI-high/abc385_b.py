H, W, X, Y = map(int, input().split())
grid = [input().strip() for _ in range(H)]
T = input().strip()

current_x = X - 1
current_y = Y - 1
visited = set()

dir_map = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

for c in T:
    dx, dy = dir_map[c]
    new_x = current_x + dx
    new_y = current_y + dy
    
    if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
        current_x, current_y = new_x, new_y
    
    if grid[current_x][current_y] == '@':
        visited.add((current_x, current_y))

print(current_x + 1, current_y + 1, len(visited))