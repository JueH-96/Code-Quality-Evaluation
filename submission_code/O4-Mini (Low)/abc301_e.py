import sys
import threading
from collections import deque

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    H, W, T = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]

    # Gather positions: start, candies, goal
    start = None
    goal = None
    candies = []
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == 'S':
                start = (i, j)
            elif c == 'G':
                goal = (i, j)
            elif c == 'o':
                candies.append((i, j))

    # Build the list of special nodes: [start] + candies + [goal]
    nodes = [start] + candies + [goal]
    N = len(nodes)  # = K + 2

    # Precompute pairwise distances by BFS on the grid
    INF = 10**18
    dist = [[INF]*N for _ in range(N)]
    # Offsets for 4 directions
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for idx in range(N):
        # BFS from nodes[idx]
        sx, sy = nodes[idx]
        dmap = [[-1]*W for _ in range(H)]
        dq = deque()
        dq.append((sx, sy))
        dmap[sx][sy] = 0
        while dq:
            x, y = dq.popleft()
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and dmap[nx][ny] < 0:
                    dmap[nx][ny] = dmap[x][y] + 1
                    dq.append((nx, ny))
        # extract distances to all special nodes
        for jdx in range(N):
            tx, ty = nodes[jdx]
            if dmap[tx][ty] >= 0:
                dist[idx][jdx] = dmap[tx][ty]

    # Number of candies
    K = len(candies)
    # dp[mask][i]: minimum moves to start->visit candies in mask, ending at node index i
    # i runs 0..K (0 is start, 1..K candies)
    # mask is K-bit mask, bit j corresponds to candies[j]
    ALL = 1 << K
    INF_DP = 10**18
    dp = [ [INF_DP]*(K+1) for _ in range(ALL) ]
    # start with no candies collected, at start (index 0)
    dp[0][0] = 0

    # Fill DP
    for mask in range(ALL):
        for i in range(K+1):
            curd = dp[mask][i]
            if curd >= INF_DP:
                continue
            # Try to go to any candy j not yet collected
            for j in range(K):
                bit = 1 << j
                if mask & bit:
                    continue
                # from node i to candy node j+1
                d_ij = dist[i][j+1]
                if d_ij >= INF:
                    continue
                newm = mask | bit
                newd = curd + d_ij
                if newd < dp[newm][j+1]:
                    dp[newm][j+1] = newd

    # Finally try to go to goal (index K+1) from some end i, check <= T
    ans = -1
    for mask in range(ALL):
        # popcount
        cnt = mask.bit_count()
        for i in range(K+1):
            d_pi = dp[mask][i]
            if d_pi >= INF_DP:
                continue
            d_to_goal = dist[i][K+1]
            if d_to_goal >= INF:
                continue
            total = d_pi + d_to_goal
            if total <= T:
                if cnt > ans:
                    ans = cnt

    print(ans)

if __name__ == "__main__":
    main()