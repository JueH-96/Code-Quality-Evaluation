import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data:
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
    points = [start] + candies + [goal]
    goal_index = k+1
    
    dr = [0, 0, 1, -1]
    dc = [1, -1, 0, 0]
    
    INF_VAL = 10**10
    dists = [[INF_VAL] * (k+2) for _ in range(k+2)]
    
    for idx in range(len(points)):
        r0, c0 = points[idx]
        dist_grid = [[INF_VAL] * W for _ in range(H)]
        q = deque()
        dist_grid[r0][c0] = 0
        q.append((r0, c0))
        while q:
            r, c = q.popleft()
            for d in range(4):
                nr = r + dr[d]
                nc = c + dc[d]
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] != '#' and dist_grid[nr][nc] == INF_VAL:
                        dist_grid[nr][nc] = dist_grid[r][c] + 1
                        q.append((nr, nc))
        for j in range(len(points)):
            rj, cj = points[j]
            dists[idx][j] = dist_grid[rj][cj]
            
    if dists[0][goal_index] > T:
        print(-1)
        return
        
    nstates = 1 << k
    dp = [[INF_VAL] * (k+1) for _ in range(nstates)]
    dp[0][0] = 0
    max_candies = -1
    
    for mask in range(nstates):
        for i in range(0, k+1):
            if dp[mask][i] == INF_VAL:
                continue
            total_moves = dp[mask][i] + dists[i][goal_index]
            if total_moves <= T:
                cnt = bin(mask).count('1')
                if cnt > max_candies:
                    max_candies = cnt
            for j in range(1, k+1):
                if mask & (1 << (j-1)):
                    continue
                new_mask = mask | (1 << (j-1))
                new_cost = dp[mask][i] + dists[i][j]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
                    
    if max_candies == -1:
        print(0)
    else:
        print(max_candies)

if __name__ == '__main__':
    main()