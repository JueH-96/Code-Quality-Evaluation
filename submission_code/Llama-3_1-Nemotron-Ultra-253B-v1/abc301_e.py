import sys
from collections import deque

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    s_pos = None
    g_pos = None
    candies = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                s_pos = (i, j)
            elif grid[i][j] == 'G':
                g_pos = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    m = len(candies)
    nodes = [s_pos] + candies + [g_pos]
    node_count = m + 2  # s, candies, g
    
    def bfs(start_i, start_j):
        INF = float('inf')
        dist = [[INF] * W for _ in range(H)]
        q = deque()
        dist[start_i][start_j] = 0
        q.append((start_i, start_j))
        while q:
            i, j = q.popleft()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] != '#' and dist[ni][nj] == INF:
                        dist[ni][nj] = dist[i][j] + 1
                        q.append((ni, nj))
        return dist
    
    dist_matrix = [[float('inf')] * node_count for _ in range(node_count)]
    for i in range(node_count):
        x, y = nodes[i]
        dist = bfs(x, y)
        for j in range(node_count):
            nx, ny = nodes[j]
            dist_matrix[i][j] = dist[nx][ny]
    
    if dist_matrix[0][node_count - 1] > T:
        print(-1)
        return
    
    INF = float('inf')
    dp = [[INF] * (m + 1) for _ in range(1 << m)]
    dp[0][0] = 0
    max_candies = -1
    
    for mask in range(0, 1 << m):
        for u in range(m + 1):
            if u == 0:
                if mask != 0:
                    continue
            else:
                if not (mask & (1 << (u - 1))):
                    continue
            current_steps = dp[mask][u]
            if current_steps == INF:
                continue
            
            d_to_g = dist_matrix[u][node_count - 1]
            if d_to_g != INF:
                total = current_steps + d_to_g
                if total <= T:
                    cnt = bin(mask).count('1')
                    if cnt > max_candies:
                        max_candies = cnt
            
            for j in range(m):
                if not (mask & (1 << j)):
                    v = j + 1
                    d = dist_matrix[u][v]
                    if d != INF:
                        new_mask = mask | (1 << j)
                        new_steps = current_steps + d
                        if new_steps < dp[new_mask][v]:
                            dp[new_mask][v] = new_steps
    
    print(max_candies if max_candies != -1 else -1)

if __name__ == '__main__':
    main()