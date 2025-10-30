# Read input
H, W, X, Y = map(int, input().split())
x = X - 1  # Convert to 0-based index
y = Y - 1

grid = [input().strip() for _ in range(H)]
T = input().strip()

current_x, current_y = x, y
houses = set()

for c in T:
    new_x, new_y = current_x, current_y
    if c == 'U':
        new_x -= 1
    elif c == 'D':
        new_x += 1
    elif c == 'L':
        new_y -= 1
    elif c == 'R':
        new_y += 1
    
    # Check if new position is within grid and passable
    if 0 <= new_x < H and 0 <= new_y < W:
        if grid[new_x][new_y] != '#':
            current_x, current_y = new_x, new_y
    
    # Add current position to houses if it's a house
    if grid[current_x][current_y] == '@':
        houses.add((current_x, current_y))

# Output the results
print(current_x + 1, current_y + 1, len(houses))