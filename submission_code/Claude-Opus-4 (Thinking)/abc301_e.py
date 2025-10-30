from collections import deque

H, W, T = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find start, goal, and candy positions
start = None
goal = None
candies = []

for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'G':
            goal = (i, j)
        elif grid[i][j] == 'o':
            candies.append((i, j))

n_candies = len(candies)

# BFS to find shortest distances from a given position
def bfs(start_pos):
    dist = [[-1] * W for _ in range(H)]
    queue = deque([start_pos])
    dist[start_pos[0]][start_pos[1]] = 0
    
    while queue:
        i, j = queue.popleft()
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#' and dist[ni][nj] == -1:
                dist[ni][nj] = dist[i][j] + 1
                queue.append((ni, nj))
    
    return dist

# Create list of all important points: [start] + candies + [goal]
all_points = [start] + candies + [goal]
n_points = len(all_points)

# Build distance matrix between all important points
INF = float('inf')
dist_matrix = [[INF] * n_points for _ in range(n_points)]

for i in range(n_points):
    dist_from_i = bfs(all_points[i])
    for j in range(n_points):
        if i == j:
            dist_matrix[i][j] = 0
        else:
            d = dist_from_i[all_points[j][0]][all_points[j][1]]
            if d != -1:
                dist_matrix[i][j] = d

# Check if goal is reachable from start within T moves
if dist_matrix[0][n_points-1] > T:
    print(-1)
else:
    if n_candies == 0:
        print(0)
    else:
        # DP with bitmask
        # dp[mask][pos] = minimum moves to visit candies in mask and end at pos
        # pos indices: 0 = start, 1..n_candies = candies, n_candies+1 = goal
        dp = [[INF] * n_points for _ in range(1 << n_candies)]
        dp[0][0] = 0  # Start at position 0 (start) with no candies visited
        
        for mask in range(1 << n_candies):
            for pos in range(n_points):
                if dp[mask][pos] >= INF:
                    continue
                
                # Try to move to goal
                goal_idx = n_points - 1
                new_dist = dp[mask][pos] + dist_matrix[pos][goal_idx]
                if new_dist <= T:
                    dp[mask][goal_idx] = min(dp[mask][goal_idx], new_dist)
                
                # Try to move to unvisited candies
                for i in range(n_candies):
                    if not (mask & (1 << i)):  # Candy i not visited
                        new_mask = mask | (1 << i)
                        candy_idx = i + 1
                        new_dist = dp[mask][pos] + dist_matrix[pos][candy_idx]
                        if new_dist <= T:
                            dp[new_mask][candy_idx] = min(dp[new_mask][candy_idx], new_dist)
        
        # Find maximum candies visited while ending at goal
        max_candies = 0
        goal_idx = n_points - 1
        for mask in range(1 << n_candies):
            if dp[mask][goal_idx] <= T:
                max_candies = max(max_candies, bin(mask).count('1'))
        
        print(max_candies)