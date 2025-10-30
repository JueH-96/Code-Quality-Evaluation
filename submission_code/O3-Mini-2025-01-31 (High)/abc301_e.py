def main():
    import sys
    from collections import deque
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # Parse inputs.
    H, W, T = map(int, data[0].split())
    grid = data[1:]
    
    # Locate start (S), goal (G) and candy cells (o).
    start = None
    goal = None
    candies = []
    for i in range(H):
        for j, ch in enumerate(grid[i]):
            if ch == 'S':
                start = (i, j)
            elif ch == 'G':
                goal = (i, j)
            elif ch == 'o':
                candies.append((i, j))
    
    # Build the list of important nodes:
    # Index 0: start, indices 1..N: candy cells, index N+1: goal.
    nodes = [start] + candies + [goal]
    n_nodes = len(nodes)
    N = len(candies)  # Number of candy nodes

    # Set a sufficiently large INF value.
    INF = 10**9

    # We'll compute pairwise shortest distances between these nodes using BFS.
    dist_mat = [[INF] * n_nodes for _ in range(n_nodes)]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    def bfs_from(si, sj):
        # Returns an HxW array of distances from (si, sj) to every cell.
        dist_arr = [[-1] * W for _ in range(H)]
        dq = deque()
        dq.append((si, sj))
        dist_arr[si][sj] = 0
        while dq:
            ci, cj = dq.popleft()
            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] != '#' and dist_arr[ni][nj] == -1:
                        dist_arr[ni][nj] = dist_arr[ci][cj] + 1
                        dq.append((ni, nj))
        return dist_arr

    # For each node, run BFS and record distances to each other node.
    for i in range(n_nodes):
        si, sj = nodes[i]
        d_arr = bfs_from(si, sj)
        for j in range(n_nodes):
            ti, tj = nodes[j]
            if d_arr[ti][tj] != -1:
                dist_mat[i][j] = d_arr[ti][tj]
            else:
                dist_mat[i][j] = INF

    # If the direct route from S to G takes more than T moves, it's impossible.
    if dist_mat[0][n_nodes - 1] > T:
        print(-1)
        return

    # Use DP (bitmask TSP) to try all orders of visiting candy nodes.
    # dp[mask][i]: minimum number of moves to reach candy i (node index i+1) while having visited the candy subset 'mask'
    dp = [[INF] * N for _ in range(1 << N)]
    
    # Base: go from start (node 0) to each candy i.
    for i in range(N):
        d = dist_mat[0][i + 1]
        if d < INF:
            dp[1 << i][i] = d

    # Iterate over all subsets of candies and update the DP table.
    for mask in range(1 << N):
        for i in range(N):
            if dp[mask][i] == INF:
                continue
            for j in range(N):
                if mask & (1 << j): 
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + dist_mat[i + 1][j + 1]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost

    # Evaluate all possibilities: from a state (possibly no candy visited) finish at goal.
    ans = -1
    # Option of not collecting any candy.
    if dist_mat[0][n_nodes - 1] <= T:
        ans = 0
    # Option with at least one candy visited.
    for mask in range(1 << N):
        cnt = bin(mask).count("1")
        for i in range(N):
            if dp[mask][i] == INF:
                continue
            total_cost = dp[mask][i] + dist_mat[i + 1][n_nodes - 1]
            if total_cost <= T:
                ans = max(ans, cnt)
    print(ans)

if __name__ == '__main__':
    main()