import sys
from collections import deque

# Read input
H, W, D = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(H)]

# Initialize distance matrix
# Use float('inf') to represent unvisited cells.
# Distance from a humidifier is 0.
dist = [[float('inf')] * W for _ in range(H)]

# Queue for BFS
queue = deque()

# Find initial humidifiers and add to queue. Set their distance to 0.
for r in range(H):
    for c in range(W):
        if grid[r][c] == 'H':
            dist[r][c] = 0
            queue.append((r, c))

# BFS directions (up, down, left, right)
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]

# Perform BFS to find shortest distance from any humidifier to all reachable cells
while queue:
    r, c = queue.popleft()

    # Optimization: If the current cell is already at or beyond the maximum
    # distance D, any path extending from this cell will exceed D steps.
    # So, we don't need to explore neighbors from this cell.
    # Cells at distance D themselves *are* counted later, but we don't expand from them
    # to find cells at distance D+1 (which are not within the limit D).
    if dist[r][c] >= D:
        continue

    # Explore neighbors
    for i in range(4):
        nr = r + dr[i]
        nc = c + dc[i]

        # Check bounds
        if 0 <= nr < H and 0 <= nc < W:
            # Check if the neighbor is not a wall and hasn't been visited yet
            # or found via a shorter path. BFS guarantees shortest path first,
            # so if dist[nr][nc] is still inf, it's the first time we're reaching it.
            if grid[nr][nc] != '#' and dist[nr][nc] == float('inf'):
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))

# Count humidified floor cells
# A cell is humidified if its shortest distance from any humidifier is <= D.
# Humidified *floor* cells are those that are '.' or 'H' AND humidified.
# The condition grid[r][c] != '#' covers both '.' and 'H' cells, excluding walls.
humidified_floor_count = 0
for r in range(H):
    for c in range(W):
        if grid[r][c] != '#' and dist[r][c] <= D:
            humidified_floor_count += 1

# Print the result
print(humidified_floor_count)