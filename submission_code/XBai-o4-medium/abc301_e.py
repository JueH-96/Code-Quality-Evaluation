import sys
from collections import deque

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    s_pos = None
    g_pos = None
    o_list = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                s_pos = (i, j)
            elif grid[i][j] == 'G':
                g_pos = (i, j)
            elif grid[i][j] == 'o':
                o_list.append((i, j))
    K = len(o_list)
    points = [s_pos] + o_list + [g_pos]
    N_points = len(points)
    
    def bfs(x, y):
        dist = [[-1] * W for _ in range(H)]
        q = deque()
        if grid[x][y] == '#':
            return dist
        dist[x][y] = 0
        q.append((x, y))
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] == '#':
                        continue
                    if dist[nx][ny] == -1:
                        dist[nx][ny] = dist[cx][cy] + 1
                        q.append((nx, ny))
        return dist
    
    dist_between = [[-1] * N_points for _ in range(N_points)]
    for i in range(N_points):
        x_i, y_i = points[i]
        bfs_dist = bfs(x_i, y_i)
        for j in range(N_points):
            x_j, y_j = points[j]
            dist_between[i][j] = bfs_dist[x_j][y_j]
    
    if dist_between[0][N_points - 1] == -1:
        print(-1)
        return
    
    max_mask = 1 << K
    INF = float('inf')
    dp = [[INF] * (K + 1) for _ in range(max_mask)]
    dp[0][0] = 0
    
    for mask in range(max_mask):
        for u in range(K + 1):
            if dp[mask][u] == INF:
                continue
            for v in range(1, K + 1):
                if not (mask & (1 << (v - 1))):
                    d = dist_between[u][v]
                    if d != -1:
                        new_mask = mask | (1 << (v - 1))
                        new_step = dp[mask][u] + d
                        if new_step < dp[new_mask][v]:
                            dp[new_mask][v] = new_step
    
    max_candies = 0
    for mask in range(max_mask):
        for u in range(K + 1):
            if dp[mask][u] == INF:
                continue
            d_to_G = dist_between[u][N_points - 1]
            if d_to_G != -1:
                total_step = dp[mask][u] + d_to_G
                if total_step <= T:
                    cnt = bin(mask).count('1')
                    if cnt > max_candies:
                        max_candies = cnt
    print(max_candies)

if __name__ == '__main__':
    main()