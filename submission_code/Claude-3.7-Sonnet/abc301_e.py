from collections import deque

# Read input
H, W, T = map(int, input().split())
grid = [input() for _ in range(H)]

# Find the start, goal, and candy positions
points = []
start_idx, goal_idx = -1, -1
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start_idx = len(points)
            points.append((i, j))
        elif grid[i][j] == 'G':
            goal_idx = len(points)
            points.append((i, j))
        elif grid[i][j] == 'o':
            points.append((i, j))

# Number of points (including start, goal, and candies)
N = len(points)

# Compute the minimum distance between all pairs of points using BFS
def bfs(start_i, start_j):
    dist = [[-1 for _ in range(W)] for _ in range(H)]
    dist[start_i][start_j] = 0
    queue = deque([(start_i, start_j)])
    while queue:
        i, j = queue.popleft()
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#' and dist[ni][nj] == -1:
                dist[ni][nj] = dist[i][j] + 1
                queue.append((ni, nj))
    return dist

# Compute the distance between all pairs of points
dist = [[float('inf') for _ in range(N)] for _ in range(N)]
for idx, (i, j) in enumerate(points):
    distances = bfs(i, j)
    for other_idx, (oi, oj) in enumerate(points):
        if distances[oi][oj] == -1:
            dist[idx][other_idx] = float('inf')
        else:
            dist[idx][other_idx] = distances[oi][oj]

# Check if the goal is reachable from the start
if dist[start_idx][goal_idx] == float('inf'):
    print(-1)
    exit()

# Check if the goal can be reached from the start within T moves
if dist[start_idx][goal_idx] > T:
    print(-1)
    exit()

# Dynamic programming to find the minimum moves
# dp[mask][last] = min moves to visit all points in mask and end at 'last'
dp = [[float('inf') for _ in range(N)] for _ in range(1 << N)]
dp[1 << start_idx][start_idx] = 0

for mask in range(1 << N):
    if (mask & (1 << start_idx)) == 0:  # Skip masks that don't include the start
        continue
    for last in range(N):
        if dp[mask][last] == float('inf') or (mask & (1 << last)) == 0:
            continue
        for next_point in range(N):
            if mask & (1 << next_point):
                continue
            if dist[last][next_point] == float('inf'):  # If the next point is not reachable from the current one
                continue
            next_mask = mask | (1 << next_point)
            dp[next_mask][next_point] = min(dp[next_mask][next_point], dp[mask][last] + dist[last][next_point])

# Find the maximum number of candies within the constraint of T moves
max_candies = 0  # If no candies can be visited, the answer is 0
for mask in range(1 << N):
    if (mask & (1 << goal_idx)) == 0 or (mask & (1 << start_idx)) == 0:  # Skip masks that don't include both the start and goal
        continue
    if dp[mask][goal_idx] <= T:
        # Count the number of candies in the mask
        candies = 0
        for i in range(N):
            if i != start_idx and i != goal_idx and (mask & (1 << i)):
                candies += 1
        max_candies = max(max_candies, candies)

print(max_candies)