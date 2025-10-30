# YOUR CODE HERE
from collections import deque

def bfs(start, goal, grid, H, W):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([(start[0], start[1], 0)])
    visited = [[False] * W for _ in range(H)]
    visited[start[0]][start[1]] = True
    
    while queue:
        x, y, dist = queue.popleft()
        if (x, y) == goal:
            return dist
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] != '#':
                visited[nx][ny] = True
                queue.append((nx, ny, dist + 1))
    return float('inf')

def max_candies(H, W, T, grid):
    start = goal = None
    candies = []
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    if not start or not goal:
        return -1
    
    all_points = [start] + candies + [goal]
    n = len(all_points)
    
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                dist[i][j] = 0
            else:
                d = bfs(all_points[i], all_points[j], grid, H, W)
                dist[i][j] = dist[j][i] = d
    
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0
    
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                dp[mask | (1 << j)][j] = min(dp[mask | (1 << j)][j], dp[mask][i] + dist[i][j])
    
    max_candies = -1
    for mask in range(1 << n):
        if dp[mask][n-1] <= T:
            max_candies = max(max_candies, bin(mask).count('1') - 2)
    
    return max_candies

import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
T = int(data[2])
grid = data[3:]

result = max_candies(H, W, T, grid)
print(result)