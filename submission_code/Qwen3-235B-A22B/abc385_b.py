h, w, x, y = map(int, input().split())
grid = [input().strip() for _ in range(h)]
t = input().strip()

current_x = x - 1
current_y = y - 1

visited = set()
visited.add((current_x, current_y))

for move in t:
    dx, dy = 0, 0
    if move == 'U':
        dx = -1
    elif move == 'D':
        dx = 1
    elif move == 'L':
        dy = -1
    elif move == 'R':
        dy = 1

    new_x = current_x + dx
    new_y = current_y + dy

    if 0 <= new_x < h and 0 <= new_y < w:
        if grid[new_x][new_y] != '#':
            current_x, current_y = new_x, new_y

    visited.add((current_x, current_y))

house_count = 0
for (i, j) in visited:
    if grid[i][j] == '@':
        house_count += 1

print(current_x + 1, current_y + 1, house_count)