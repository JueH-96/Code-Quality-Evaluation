import sys
from collections import deque

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = []
    sx = sy = gx = gy = -1
    candies = []
    for i in range(H):
        line = sys.stdin.readline().strip()
        grid.append(line)
        for j in range(W):
            if line[j] == 'S':
                sx, sy = i, j
            elif line[j] == 'G':
                gx, gy = i, j
            elif line[j] == 'o':
                candies.append((i, j))
    
    # Nodes: S (0), G (1), then candies (2, 3, ...)
    nodes = [(sx, sy), (gx, gy)] + candies
    C = len(candies)
    n_nodes = len(nodes)
    
    # Precompute distance between all pairs of nodes using BFS
    INF = 1 << 60
    dist = [[INF] * n_nodes for _ in range(n_nodes)]
    
    def bfs(x, y):
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        visited = [[-1] * W for _ in range(H)]
        q = deque()
        q.append((x, y))
        visited[x][y] = 0
        while q:
            cx, cy = q.popleft()
            for d in range(4):
                nx = cx + dx[d]
                ny = cy + dy[d]
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] != '#' and visited[nx][ny] == -1:
                        visited[nx][ny] = visited[cx][cy] + 1
                        q.append((nx, ny))
        return visited
    
    for i in range(n_nodes):
        x, y = nodes[i]
        visited = bfs(x, y)
        for j in range(n_nodes):
            a, b = nodes[j]
            if visited[a][b] != -1:
                dist[i][j] = visited[a][b]
            else:
                dist[i][j] = INF
    
    if C == 0:
        if dist[0][1] <= T:
            print(0)
        else:
            print(-1)
        return
    
    size = 1 << C
    dp = [[INF] * n_nodes for _ in range(size)]
    dp[0][0] = 0  # start at S (0) with 0 candies collected
    
    for mask in range(size):
        for u in range(n_nodes):
            if dp[mask][u] > T:
                continue
            for v in range(n_nodes):
                if u == v:
                    continue
                if dist[u][v] == INF:
                    continue
                new_steps = dp[mask][u] + dist[u][v]
                if new_steps > T:
                    continue
                if v == 1:
                    new_mask = mask
                elif v >= 2:
                    candy_idx = v - 2
                    if (mask & (1 << candy_idx)) == 0:
                        new_mask = mask | (1 << candy_idx)
                    else:
                        new_mask = mask
                else:
                    new_mask = mask
                if new_steps < dp[new_mask][v]:
                    dp[new_mask][v] = new_steps
    
    max_candies = -1
    for mask in range(size):
        if dp[mask][1] <= T:
            cnt = bin(mask).count('1')
            if cnt > max_candies:
                max_candies = cnt
    
    print(max_candies if max_candies != -1 else -1)

if __name__ == "__main__":
    main()