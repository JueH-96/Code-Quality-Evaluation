from collections import deque

# Read input
H, W, D = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Directions for moving up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Initialize a queue for BFS and a set to keep track of visited cells
queue = deque()
visited = set()

# Find all humidifier cells and add them to the queue
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            queue.append((i, j, 0))  # (row, column, distance)
            visited.add((i, j))

# Perform BFS to find all humidified cells
while queue:
    x, y, dist = queue.popleft()
    if dist > D:
        continue
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited and grid[nx][ny] != '#':
            visited.add((nx, ny))
            queue.append((nx, ny, dist + 1))

# Count the number of humidified floor cells
humidified_count = sum(1 for i in range(H) for j in range(W) if grid[i][j] == '.' and (i, j) in visited)

# Print the result
print(humidified_count)