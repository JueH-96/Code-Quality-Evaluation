from collections import deque

# Read input
H, W, D = map(int, input().split())
grid = [input() for _ in range(H)]

# Define directions: up, right, down, left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Track which cells are humidified
humidified = [[False for _ in range(W)] for _ in range(H)]

# Queue for BFS
queue = deque()

# Find all humidifier cells and add them to the queue
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'H':
            queue.append((i, j, 0))  # (row, col, distance)
            humidified[i][j] = True

# BFS to find all humidified cells
while queue:
    i, j, dist = queue.popleft()
    
    # Check all four directions
    for di, dj in directions:
        ni, nj = i + di, j + dj
        
        # Check if the new position is valid and the distance is within D
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#' and not humidified[ni][nj] and dist < D:
            humidified[ni][nj] = True
            queue.append((ni, nj, dist + 1))

# Count the number of humidified floor cells
count = 0
for i in range(H):
    for j in range(W):
        if humidified[i][j] and (grid[i][j] == '.' or grid[i][j] == 'H'):
            count += 1

# Print the answer
print(count)