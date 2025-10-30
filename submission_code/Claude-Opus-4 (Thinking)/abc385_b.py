# YOUR CODE HERE
# Read input
H, W, X, Y = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())
T = input().strip()

# Convert to 0-indexed for easier array access
X -= 1
Y -= 1

# Track current position
curr_x, curr_y = X, Y

# Track houses visited (using set for uniqueness)
houses_visited = set()

# Process each move in T
for move in T:
    next_x, next_y = curr_x, curr_y
    
    if move == 'U':
        next_x = curr_x - 1
    elif move == 'D':
        next_x = curr_x + 1
    elif move == 'L':
        next_y = curr_y - 1
    elif move == 'R':
        next_y = curr_y + 1
    
    # Check if next position is valid and passable
    if 0 <= next_x < H and 0 <= next_y < W and grid[next_x][next_y] != '#':
        curr_x, curr_y = next_x, next_y
        
        # Check if new position has a house
        if grid[curr_x][curr_y] == '@':
            houses_visited.add((curr_x, curr_y))

# Convert back to 1-indexed for output
print(curr_x + 1, curr_y + 1, len(houses_visited))