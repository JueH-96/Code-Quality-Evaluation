H, W, X, Y = map(int, input().split())
grid = [input().strip() for _ in range(H)]
T = input().strip()

current_x, current_y = X, Y
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
    
    valid = False
    if 1 <= next_x <= H and 1 <= next_y <= W:
        cell = grid[next_x - 1][next_y - 1]
        if cell != '#':
            valid = True
    
    if valid:
        current_x, current_y = next_x, next_y
    
    # Check current cell after moving
    cell_current = grid[current_x - 1][current_y - 1]
    if cell_current == '@':
        houses.add((current_x, current_y))

print(current_x, current_y, len(houses))