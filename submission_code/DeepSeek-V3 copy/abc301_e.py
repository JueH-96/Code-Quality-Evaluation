from collections import deque
import sys
import itertools

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Find positions of S, G, and candies
    S = G = None
    candies = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                S = (i, j)
            elif grid[i][j] == 'G':
                G = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    # Precompute distances between all pairs of positions
    positions = [S] + candies + [G]
    n = len(positions)
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        start = positions[i]
        q = deque()
        q.append((start[0], start[1], 0))
        visited = [[-1] * W for _ in range(H)]
        visited[start[0]][start[1]] = 0
        while q:
            x, y, steps = q.popleft()
            if steps > T:
                break
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and visited[nx][ny] == -1:
                    visited[nx][ny] = steps + 1
                    q.append((nx, ny, steps + 1))
        for j in range(n):
            target = positions[j]
            if visited[target[0]][target[1]] != -1:
                dist[i][j] = visited[target[0]][target[1]]
    
    # Check if it's possible to reach G from S within T moves
    if dist[0][n-1] > T:
        print(-1)
        return
    
    # Now, find the maximum number of candies that can be collected
    # We have to find a path from S to G that visits as many candies as possible, with total moves <= T
    # Since there are at most 18 candies, we can use bitmask DP
    # dp[mask][u] = minimum steps to reach u with mask of collected candies
    # mask is a bitmask representing which candies have been collected
    # u is the index in positions
    
    # Initialize DP
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << len(candies))]
    dp[0][0] = 0
    
    # Iterate over all masks
    for mask in range(1 << len(candies)):
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            for v in range(n):
                if dist[u][v] == INF:
                    continue
                new_mask = mask
                if v > 0 and v <= len(candies):
                    new_mask |= (1 << (v-1))
                if dp[new_mask][v] > dp[mask][u] + dist[u][v]:
                    dp[new_mask][v] = dp[mask][u] + dist[u][v]
    
    # Now, find the mask with the maximum number of candies, where dp[mask][n-1] <= T
    max_candies = 0
    for mask in range(1 << len(candies)):
        if dp[mask][n-1] <= T:
            cnt = bin(mask).count('1')
            if cnt > max_candies:
                max_candies = cnt
    
    print(max_candies)

if __name__ == "__main__":
    main()