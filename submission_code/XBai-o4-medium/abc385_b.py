H, W, X, Y = map(int, input().split())
grid = [input().strip() for _ in range(H)]
T = input().strip()

current_x, current_y = X, Y
houses = set()

for c in T:
    dx = 0
    dy = 0
    if c == 'U':
        dx = -1
    elif c == 'D':
        dx = 1
    elif c == 'L':
        dy = -1
    elif c == 'R':
        dy = 1
    
    next_x = current_x + dx
    next_y = current_y + dy
    
    # Check if the next position is within the grid and not a wall
    if 1 <= next_x <= H and 1 <= next_y <= W:
        if grid[next_x - 1][next_y - 1] != '#':
            current_x, current_y = next_x, next_y
    
    # Check if current cell is a house
    cell = grid[current_x - 1][current_y - 1]
    if cell == '@':
        houses.add((current_x, current_y))

print(current_x, current_y, len(houses))