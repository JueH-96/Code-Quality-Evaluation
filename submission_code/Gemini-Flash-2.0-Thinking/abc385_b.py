H, W, X, Y = map(int, input().split())
grid = [input() for _ in range(H)]
T = input()

current_x = X - 1
current_y = Y - 1

visited_houses = set()

if grid[current_x][current_y] == '@':
    visited_houses.add((current_x, current_y))

for move in T:
    next_x, next_y = current_x, current_y

    if move == 'U':
        next_x -= 1
    elif move == 'D':
        next_x += 1
    elif move == 'L':
        next_y -= 1
    elif move == 'R':
        next_y += 1

    if 0 <= next_x < H and 0 <= next_y < W and grid[next_x][next_y] != '#':
        current_x, current_y = next_x, next_y
        if grid[current_x][current_y] == '@':
            visited_houses.add((current_x, current_y))

print(current_x + 1, current_y + 1, len(visited_houses))