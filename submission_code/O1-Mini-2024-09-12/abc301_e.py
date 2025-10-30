# YOUR CODE HERE
import sys
import sys
import sys
from collections import deque

def main():
    import sys
    import sys
    H, W, T = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())

    positions = []
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
    n = len(candies)
    nodes = [S] + candies + [G]
    node_indices = {pos: idx for idx, pos in enumerate(nodes)}
    num_nodes = len(nodes)
    INF = 1 << 60
    # Compute pairwise distances
    distances = [[INF]*num_nodes for _ in range(num_nodes)]
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    def bfs(start_idx):
        dist = [[-1]*W for _ in range(H)]
        q = deque()
        si, sj = nodes[start_idx]
        dist[si][sj] = 0
        q.append((si, sj))
        while q:
            i, j = q.popleft()
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] != '#' and dist[ni][nj]==-1:
                        dist[ni][nj] = dist[i][j] + 1
                        q.append((ni, nj))
        for end_idx in range(num_nodes):
            ei, ej = nodes[end_idx]
            if dist[ei][ej] != -1:
                distances[start_idx][end_idx] = dist[ei][ej]
    for idx in range(num_nodes):
        bfs(idx)
    S_idx = 0
    G_idx = num_nodes -1
    candy_indices = list(range(1, num_nodes-1))
    # Check if S to G is reachable
    if distances[S_idx][G_idx] > T:
        print(-1)
        return
    # DP[mask][last] = min distance
    SIZE = 1 << n
    DP = [ [INF]*num_nodes for _ in range(SIZE)]
    DP[0][S_idx] = 0
    for mask in range(SIZE):
        for last in range(num_nodes):
            if DP[mask][last] <= T:
                for c in range(n):
                    if not (mask & (1 << c)):
                        next_mask = mask | (1 << c)
                        to = c+1
                        new_dist = DP[mask][last] + distances[last][to]
                        if distances[last][to] != INF and new_dist < DP[next_mask][to]:
                            DP[next_mask][to] = new_dist
    max_candies = -1
    for mask in range(SIZE):
        cnt = bin(mask).count('1')
        for last in range(num_nodes):
            if DP[mask][last] + distances[last][G_idx] <= T:
                if cnt > max_candies:
                    max_candies = cnt
    print(max_candies if max_candies != -1 else -1)

if __name__ == "__main__":
    main()