# Read the input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Function to check if a cell is a sensor
def is_sensor(i, j):
    return grid[i][j] == '#'

# Function to check if two cells are adjacent
def are_adjacent(i1, j1, i2, j2):
    return max(abs(i1 - i2), abs(j1 - j2)) == 1

# Perform a depth-first search to find connected sensors
def dfs(i, j, visited):
    visited.add((i, j))
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if are_adjacent(i, j, i + di, j + dj) and is_sensor(i + di, j + dj) and (i + di, j + dj) not in visited:
                dfs(i + di, j + dj, visited)

# Count the number of sensors
sensors = 0
visited = set()
for i in range(H):
    for j in range(W):
        if is_sensor(i, j) and (i, j) not in visited:
            sensors += 1
            dfs(i, j, visited)

# Print the answer
print(sensors)