# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    import heapq
    sys.setrecursionlimit(1 << 25)
    H, W, T = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())

    N = 2  # Node indices: 0 for S, 1 for G
    s_pos = None
    g_pos = None
    candy_positions = []
    candy_indices = {}
    num_candies = 0
    node_positions = []

    for i in range(H):
        for j in range(W):
            cell = grid[i][j]
            if cell == 'S':
                s_pos = (i, j)
                node_positions.append((i, j))  # index 0
            elif cell == 'G':
                g_pos = (i, j)
                node_positions.append((i, j))  # index 1
            elif cell == 'o':
                candy_positions.append((i, j))
                candy_indices[(i, j)] = num_candies  # Map position to candy index
                node_positions.append((i, j))  # Node index from N onwards
                num_candies +=1
                N +=1

    # Build dist[N][N] by BFS from each node
    H, W = len(grid), len(grid[0])
    dist = [[math.inf]*N for _ in range(N)]
    for idx in range(N):
        sx, sy = node_positions[idx]
        queue = [(sx, sy)]
        visited = [[-1]*W for _ in range(H)]
        visited[sx][sy] = 0
        head = 0
        while head < len(queue):
            x, y = queue[head]
            head +=1
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x+dx, y+dy
                if 0<=nx<H and 0<=ny<W and grid[nx][ny]!='#' and visited[nx][ny]==-1:
                    visited[nx][ny] = visited[x][y]+1
                    queue.append((nx, ny))
        # For each other node, store distance
        for jdx in range(N):
            tx, ty = node_positions[jdx]
            if visited[tx][ty]!=-1:
                dist[idx][jdx] = visited[tx][ty]
    # Initialize DP[mask][u] = INF
    num_masks = 1<<num_candies
    INF = math.inf
    DP = [ [INF]*N for _ in range(num_masks) ]
    s_idx = 0
    g_idx = 1
    DP[0][s_idx] = 0
    # Process masks in increasing order (can be optimized)
    from collections import deque
    for mask in range(num_masks):
        for u in range(N):
            if DP[mask][u] < INF:
                for v in range(N):
                    if dist[u][v]==math.inf:
                        continue
                    if u == v:
                        continue
                    new_mask = mask
                    if v >=2:  # Candy nodes
                        c = v -2
                        if not (mask & (1<<c)):
                            new_mask |= (1<<c)
                    cost = DP[mask][u]+ dist[u][v]
                    if DP[new_mask][v] > cost:
                        DP[new_mask][v] = cost
                # Update cost to G
                cost_to_G = DP[mask][u]+ dist[u][g_idx]
                if DP[mask][g_idx] > cost_to_G:
                    DP[mask][g_idx] = cost_to_G
    # For each candies_collected, find minimal DP[mask][G]
    max_candies = -1
    min_DP_G = [INF]*(num_candies+1)
    for mask in range(num_masks):
        candies_collected = bin(mask).count('1')
        if DP[mask][g_idx]<=T:
            if candies_collected > max_candies:
                max_candies = candies_collected
    if max_candies == -1:
        print(-1)
    else:
        print(max_candies)

if __name__ == "__main__":
    threading.Thread(target=main).start()