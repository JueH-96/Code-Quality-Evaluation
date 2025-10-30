# Read input
H, W, X, Y = map(int, input().split())

# Read grid
grid = []
for i in range(H):
    grid.append(input().strip())

# Read movement string
T = input().strip()

# Convert to 0-indexed coordinates
current_x = X - 1
current_y = Y - 1

# Set to track visited houses
visited_houses = set()

# Check if starting position has a house
if grid[current_x][current_y] == '@':
    visited_houses.add((current_x, current_y))

# Process each movement
for move in T:
    new_x, new_y = current_x, current_y
    
    if move == 'U':
        new_x = current_x - 1
    elif move == 'D':
        new_x = current_x + 1
    elif move == 'L':
        new_y = current_y - 1
    elif move == 'R':
        new_y = current_y + 1
    
    # Check if the new position is valid (within bounds and passable)
    if (0 <= new_x < H and 0 <= new_y < W and 
        grid[new_x][new_y] != '#'):
        # Move to new position
        current_x, current_y = new_x, new_y
        
        # Check if new position has a house
        if grid[current_x][current_y] == '@':
            visited_houses.add((current_x, current_y))

# Convert back to 1-indexed coordinates for output
final_x = current_x + 1
final_y = current_y + 1
house_count = len(visited_houses)

print(final_x, final_y, house_count)