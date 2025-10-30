# YOUR CODE HERE
# Read input
H, W, X, Y = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())
T = input().strip()

# Convert to 0-indexed for easier array access
current_x = X - 1
current_y = Y - 1

# Set to track unique houses visited
houses_visited = set()

# Check if starting position has a house (though problem says it's '.')
if grid[current_x][current_y] == '@':
    houses_visited.add((current_x, current_y))

# Process each movement command
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
    
    # Check if new position is valid and passable
    if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
        current_x = new_x
        current_y = new_y
        
        # If there's a house at the new position, add it to visited set
        if grid[current_x][current_y] == '@':
            houses_visited.add((current_x, current_y))

# Convert back to 1-indexed and output
print(current_x + 1, current_y + 1, len(houses_visited))