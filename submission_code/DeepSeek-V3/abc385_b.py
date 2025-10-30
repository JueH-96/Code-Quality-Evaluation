# YOUR CODE HERE
H, W, X, Y = map(int, input().split())
grid = [list(input().strip()) for _ in range(H)]
T = input().strip()

# Adjust to 0-based index
x, y = X-1, Y-1

# Initialize the set to keep track of visited houses
visited_houses = set()

# Check if the current cell is a house
if grid[x][y] == '@':
    visited_houses.add((x, y))

# Iterate through each move in T
for move in T:
    # Determine the new position
    if move == 'U':
        new_x, new_y = x-1, y
    elif move == 'D':
        new_x, new_y = x+1, y
    elif move == 'L':
        new_x, new_y = x, y-1
    elif move == 'R':
        new_x, new_y = x, y+1
    else:
        new_x, new_y = x, y
    
    # Check if the new position is within bounds and passable
    if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
        x, y = new_x, new_y
        # Check if the new cell is a house
        if grid[x][y] == '@':
            visited_houses.add((x, y))

# Output the final position and the count of distinct houses
print(x+1, y+1, len(visited_houses))