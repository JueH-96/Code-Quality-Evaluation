import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().rstrip()) for _ in range(H)]
    N = int(sys.stdin.readline())
    meds = {}
    # Read medicines
    for _ in range(N):
        r, c, e = map(int, sys.stdin.readline().split())
        meds[(r-1, c-1)] = e
    # Locate S and T
    start = None
    goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    # Build list of special nodes: S, each medicine cell, T
    points = [start]
    E = [0]  # energy reset values, 0 for start
    # store index mapping for medicines
    for p, e in meds.items():
        points.append(p)
        E.append(e)
    points.append(goal)
    E.append(0)  # no medicine at T
    M = len(points)  # = 2 + N
    # Precompute pairwise distances using BFS from each point
    INF = 10**9
    dist = [[INF]*M for _ in range(M)]
    # directions
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for i in range(M):
        sx, sy = points[i]
        dmap = [[-1]*W for _ in range(H)]
        dq = deque()
        dmap[sx][sy] = 0
        dq.append((sx, sy))
        while dq:
            x, y = dq.popleft()
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and dmap[nx][ny] == -1:
                    dmap[nx][ny] = dmap[x][y] + 1
                    dq.append((nx, ny))
        # fill dist[i][*]
        for j in range(M):
            tx, ty = points[j]
            if dmap[tx][ty] != -1:
                dist[i][j] = dmap[tx][ty]
    # Now do a max-energy Dijkstra-like on the graph of special points
    # dp[i] = max energy we can have upon arriving at point i (before using its med)
    dp = [-1]*(M)
    import heapq
    # max-heap via negative
    pq = []
    # start at node 0 with energy 0
    dp[0] = 0
    heapq.heappush(pq, (0, 0))  # ( -energy, node )
    target_idx = M-1
    while pq:
        ne, u = heapq.heappop(pq)
        e = -ne
        if dp[u] != e:
            continue
        # if we reached goal
        if u == target_idx:
            print("Yes")
            return
        # Optionally use medicine here to reset energy
        if E[u] > e:
            e = E[u]
            dp[u] = e
        # Try to go to all other special points
        for v in range(M):
            if v == u: continue
            d = dist[u][v]
            if d <= e:
                ne2 = e - d
                if ne2 > dp[v]:
                    dp[v] = ne2
                    heapq.heappush(pq, (-ne2, v))
    # If we exit loop without reaching T
    print("No")

if __name__ == "__main__":
    main()