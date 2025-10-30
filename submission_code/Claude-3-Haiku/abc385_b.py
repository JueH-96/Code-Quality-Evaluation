# Read the input
H, W, X, Y = map(int, input().split())
grid = [list(input()) for _ in range(H)]
T = input()

# Initialize the starting position and the set of visited houses
x, y = X-1, Y-1
visited_houses = set()

# Iterate through the actions in T
for action in T:
    if action == 'U' and x > 0 and grid[x-1][y] != '#':
        x -= 1
    elif action == 'D' and x < H-1 and grid[x+1][y] != '#':
        x += 1
    elif action == 'L' and y > 0 and grid[x][y-1] != '#':
        y -= 1
    elif action == 'R' and y < W-1 and grid[x][y+1] != '#':
        y += 1
    
    # Check if the current cell contains a house
    if grid[x][y] == '@':
        visited_houses.add((x, y))

# Print the final position and the number of distinct houses visited
print(x+1, y+1, len(visited_houses))