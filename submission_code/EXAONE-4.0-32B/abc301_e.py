import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        print(-1)
        return
        
    H, W, T = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
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
                
    k = len(candies)
    points = [start, goal] + candies
    n_nodes = len(points)
    
    if n_nodes == 0:
        print(-1)
        return
        
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    INF = 10**9
    D = [[INF] * n_nodes for _ in range(n_nodes)]
    
    for idx in range(n_nodes):
        r0, c0 = points[idx]
        dist_grid = [[-1] * W for _ in range(H)]
        q = deque()
        q.append((r0, c0))
        dist_grid[r0][c0] = 0
        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] != '#' and dist_grid[nr][nc] == -1:
                        dist_grid[nr][nc] = dist_grid[r][c] + 1
                        q.append((nr, nc))
        for j in range(n_nodes):
            rj, cj = points[j]
            d_val = dist_grid[rj][cj]
            if d_val == -1:
                D[idx][j] = INF
            else:
                D[idx][j] = d_val
                
    n_masks = 1 << k
    dp = [[INF] * n_nodes for _ in range(n_masks)]
    dp[0][0] = 0
    
    for mask in range(n_masks):
        for i in range(n_nodes):
            if dp[mask][i] == INF:
                continue
            if i == 1:
                continue
            j = 1
            d_ij = D[i][j]
            if d_ij < INF:
                new_cost = dp[mask][i] + d_ij
                if new_cost < dp[mask][j]:
                    dp[mask][j] = new_cost
                    
            for j_idx in range(2, n_nodes):
                candy_index = j_idx - 2
                if mask & (1 << candy_index):
                    continue
                d_ij = D[i][j_idx]
                if d_ij == INF:
                    continue
                new_mask = mask | (1 << candy_index)
                new_cost = dp[mask][i] + d_ij
                if new_cost < dp[new_mask][j_idx]:
                    dp[new_mask][j_idx] = new_cost
                    
    ans = -1
    for mask in range(n_masks):
        if dp[mask][1] <= T:
            cnt = bin(mask).count('1')
            if cnt > ans:
                ans = cnt
                
    print(ans if ans != -1 else -1)

if __name__ == "__main__":
    main()