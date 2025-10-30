# Read input
H, W, X, Y = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input()))
T = input()

# Set to keep track of unique house positions visited
houses_visited = set()

# Current position (convert to 0-based indexing)
curr_x = X - 1
curr_y = Y - 1

# Check if current position has a house
def check_house(x, y):
    if grid[x][y] == '@':
        houses_visited.add((x, y))

# Check if a position is passable
def is_passable(x, y):
    return grid[x][y] != '#'

# Initial position might have a house
check_house(curr_x, curr_y)

# Process each move
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
        
    # Check if next position is passable
    if is_passable(next_x, next_y):
        curr_x, curr_y = next_x, next_y
        check_house(curr_x, curr_y)

# Convert back to 1-based indexing for output
print(curr_x + 1, curr_y + 1, len(houses_visited))