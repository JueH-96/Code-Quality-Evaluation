# Read input
H, W, X, Y = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())
T = input().strip()

# Initialize position (convert to 0-indexed)
x, y = X - 1, Y - 1

# Set to track visited houses
visited_houses = set()

# Check if starting position is a house
if grid[x][y] == '@':
    visited_houses.add((x, y))

# Process each move
for move in T:
    new_x, new_y = x, y
    
    # Determine new position based on move
    if move == 'U':
        new_x = x - 1
    elif move == 'D':
        new_x = x + 1
    elif move == 'L':
        new_y = y - 1
    elif move == 'R':
        new_y = y + 1
    
    # Check if new position is valid and passable
    if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
        x, y = new_x, new_y
        # If new position is a house, add to set
        if grid[x][y] == '@':
            visited_houses.add((x, y))

# Output final position (convert back to 1-indexed) and house count
print(x + 1, y + 1, len(visited_houses))