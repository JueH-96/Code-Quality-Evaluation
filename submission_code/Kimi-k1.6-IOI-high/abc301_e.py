import sys
from collections import deque

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    S = None
    G = None
    candies = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                S = (i, j)
            elif grid[i][j] == 'G':
                G = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    K = len(candies)
    points = [S, G] + candies
    len_points = len(points)
    
    # Precompute distance matrix using BFS for each point
    dist = [[-1] * len_points for _ in range(len_points)]
    for i in range(len_points):
        start_i, start_j = points[i]
        visited = [[-1] * W for _ in range(H)]
        q = deque()
        visited[start_i][start_j] = 0
        q.append((start_i, start_j))
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] != '#' and visited[nx][ny] == -1:
                        visited[nx][ny] = visited[x][y] + 1
                        q.append((nx, ny))
        for j in range(len_points):
            xj, yj = points[j]
            dist[i][j] = visited[xj][yj]
    
    # Check if S to G is unreachable or exceeds T
    if dist[0][1] == -1:
        print(-1)
        return
    if dist[0][1] > T:
        print(-1)
        return
    
    if K == 0:
        print(0)
        return
    
    size_mask = 1 << K
    INF = float('inf')
    dp = [[INF] * len_points for _ in range(size_mask)]
    dp[0][0] = 0  # Starting at S with mask 0
    
    # Process masks in order of increasing number of bits
    masks = sorted(range(size_mask), key=lambda x: bin(x).count('1'))
    for mask in masks:
        for u in range(len_points):
            if dp[mask][u] == INF:
                continue
            for v in range(len_points):
                if u == v:
                    continue
                if dist[u][v] == -1:
                    continue
                new_time = dp[mask][u] + dist[u][v]
                # Determine new_mask and new_u
                if v >= 2:  # v is a candy
                    candy_idx = v - 2
                    if (mask & (1 << candy_idx)) == 0:
                        new_mask = mask | (1 << candy_idx)
                        new_u = v
                        if new_time < dp[new_mask][new_u]:
                            dp[new_mask][new_u] = new_time
                    else:
                        new_mask = mask
                        new_u = v
                        if new_time < dp[new_mask][new_u]:
                            dp[new_mask][new_u] = new_time
                else:
                    new_mask = mask
                    new_u = v
                    if new_time < dp[new_mask][new_u]:
                        dp[new_mask][new_u] = new_time
    
    max_candies = -1
    for mask in range(size_mask):
        cnt = bin(mask).count('1')
        for u in range(len_points):
            if dp[mask][u] != INF and dist[u][1] != -1:
                total_time = dp[mask][u] + dist[u][1]
                if total_time <= T and cnt > max_candies:
                    max_candies = cnt
    print(max_candies)

if __name__ == '__main__':
    main()