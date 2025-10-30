H, W, X, Y = map(int, input().split())
grid = [input().strip() for _ in range(H)]
T = input().strip()

current_x = X - 1
current_y = Y - 1

houses = set()

for c in T:
    next_x, next_y = current_x, current_y
    if c == 'U':
        next_x -= 1
    elif c == 'D':
        next_x += 1
    elif c == 'L':
        next_y -= 1
    elif c == 'R':
        next_y += 1
    # Check if next position is valid and not a wall
    if 0 <= next_x < H and 0 <= next_y < W and grid[next_x][next_y] != '#':
        current_x, current_y = next_x, next_y
    # Check current cell for house
    if grid[current_x][current_y] == '@':
        houses.add((current_x, current_y))

print(current_x + 1, current_y + 1, len(houses))