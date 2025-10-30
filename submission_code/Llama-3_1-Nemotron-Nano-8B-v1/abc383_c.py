import sys
from collections import deque

h, w, d = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(h)]

distance = [[d + 1] * w for _ in range(h)]
q = deque()

# Initialize queue with all H cells
for i in range(h):
    for j in range(w):
        if grid[i][j] == 'H':
            distance[i][j] = 0
            q.append((i, j))

# Directions for BFS (up, down, left, right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Perform BFS
while q:
    i, j = q.popleft()
    current_dist = distance[i][j]
    if current_dist > d:
        continue
    for dx, dy in directions:
        x, y = i + dx, j + dy
        if 0 <= x < h and 0 <= y < w:
            if grid[x][y] != '#':  # Check if not a wall
                new_dist = current_dist + 1
                if new_dist < distance[x][y] and new_dist <= d:
                    distance[x][y] = new_dist
                    q.append((x, y))

# Count humidified cells
count = 0
for i in range(h):
    for j in range(w):
        if grid[i][j] == 'H':
            count += 1
        elif grid[i][j] == '.' and distance[i][j] <= d:
            count += 1

print(count)