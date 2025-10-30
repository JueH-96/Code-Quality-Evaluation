from collections import deque
import sys

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Find start, goal, and candies
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
    
    # Precompute distances between all pairs of positions
    positions = [start] + candies + [goal]
    n = len(positions)
    dist = [[-1 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        q = deque()
        q.append((positions[i][0], positions[i][1], 0))
        visited = [[-1 for _ in range(W)] for _ in range(H)]
        visited[positions[i][0]][positions[i][1]] = 0
        while q:
            x, y, d = q.popleft()
            if d > T:
                break
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and visited[nx][ny] == -1:
                    visited[nx][ny] = d + 1
                    q.append((nx, ny, d + 1))
        for j in range(n):
            x, y = positions[j]
            dist[i][j] = visited[x][y]
    
    # Check if start to goal is possible
    if dist[0][-1] == -1 or dist[0][-1] > T:
        print(-1)
        return
    
    # Now, find the maximum number of candies
    # We can use bitmask DP to represent the set of candies visited
    # dp[mask][u] = minimum distance to reach position u with mask
    # mask represents the set of candies visited
    # Initialize dp
    INF = float('inf')
    dp = [[INF for _ in range(n)] for _ in range(1 << len(candies))]
    dp[0][0] = 0
    
    for mask in range(1 << len(candies)):
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            for v in range(n):
                if dist[u][v] == -1:
                    continue
                new_mask = mask
                if 1 <= v <= len(candies):
                    new_mask |= 1 << (v-1)
                if dp[new_mask][v] > dp[mask][u] + dist[u][v]:
                    dp[new_mask][v] = dp[mask][u] + dist[u][v]
    
    # Now, find the maximum number of candies such that dp[mask][n-1] <= T
    max_candies = 0
    for mask in range(1 << len(candies)):
        if dp[mask][-1] <= T:
            cnt = bin(mask).count('1')
            if cnt > max_candies:
                max_candies = cnt
    
    print(max_candies)

if __name__ == "__main__":
    main()