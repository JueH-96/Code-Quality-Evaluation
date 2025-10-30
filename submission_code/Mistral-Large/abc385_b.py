import sys

# Read input
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
X = int(data[2])
Y = int(data[3])

grid = []
for i in range(H):
    grid.append(data[4 + i])

T = data[4 + H]

# Initialize starting position and visited houses set
current_x, current_y = X, Y
visited_houses = set()

# Define movement directions
movements = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# Process each move in T
for move in T:
    dx, dy = movements[move]
    new_x, new_y = current_x + dx, current_y + dy

    # Check if the new position is within bounds and passable
    if 1 <= new_x <= H and 1 <= new_y <= W and grid[new_x - 1][new_y - 1] != '#':
        current_x, current_y = new_x, new_y

        # Check if the new position contains a house
        if grid[current_x - 1][current_y - 1] == '@':
            visited_houses.add((current_x, current_y))

# Output the final position and the number of distinct houses visited
print(current_x, current_y, len(visited_houses))