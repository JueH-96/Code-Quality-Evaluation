# Read input
H, W, X, Y = map(int, input().split())
T = input().strip()

# Convert to 0-based index
x = X - 1
y = Y - 1

# Read the grid
grid = [input().strip() for _ in range(H)]

visited = set()

current_x, current_y = x, y

for c in T:
    # Determine new position based on direction
    if c == 'U':
        new_x = current_x - 1
        new_y = current_y
    elif c == 'D':
        new_x = current_x + 1
        new_y = current_y
    elif c == 'L':
        new_x = current_x
        new_y = current_y - 1
    elif c == 'R':
        new_x = current_x
        new_y = current_y + 1
    else:
        continue  # invalid direction, shouldn't happen
    
    # Check if new position is within bounds and passable
    if 0 <= new_x < H and 0 <= new_y < W:
        if grid[new_x][new_y] != '#':
            current_x, current_y = new_x, new_y
            if grid[current_x][current_y] == '@':
                visited.add((current_x, current_y))

# Calculate final position and count
final_x, final_y = current_x, current_y
count = len(visited)

print(final_x + 1, final_y + 1, count)