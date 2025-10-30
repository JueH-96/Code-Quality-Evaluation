def main():
    import sys, collections
    input = sys.stdin.readline
    # Read input values
    data = sys.stdin.read().strip().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    T = int(data[2])
    grid = []
    idx = 3
    for _ in range(H):
        grid.append(data[idx])
        idx += 1

    # Find positions of S, G, and candies (o)
    start = None
    goal = None
    candies = []
    for i in range(H):
        for j in range(W):
            ch = grid[i][j]
            if ch == 'S':
                start = (i, j)
            elif ch == 'G':
                goal = (i, j)
            elif ch == 'o':
                candies.append((i, j))
    
    # Create key nodes:
    # index 0: start, indices 1..k: candies,
    # index k+1: goal.
    keys = [start] + candies + [goal]
    n = len(keys)  # total number of key nodes = k + 2, where k = len(candies)
    k = len(candies)
    
    # Pre-compute pairwise distances between key nodes using BFS.
    # d[i][j] will store the minimum number of moves from keys[i] to keys[j].
    INF = 10**9
    d = [[INF] * n for _ in range(n)]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(n):
        (si, sj) = keys[i]
        dist_grid = [[-1] * W for _ in range(H)]
        dq = collections.deque()
        dq.append((si, sj))
        dist_grid[si][sj] = 0
        while dq:
            ci, cj = dq.popleft()
            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#' and dist_grid[ni][nj] == -1:
                    dist_grid[ni][nj] = dist_grid[ci][cj] + 1
                    dq.append((ni, nj))
        for j in range(n):
            (ti, tj) = keys[j]
            if dist_grid[ti][tj] != -1:
                d[i][j] = dist_grid[ti][tj]
    
    # Use bitmask DP to decide which candies to visit.
    # Let dp[mask][v] be the minimum moves required to be at node v having visited
    # all candies in mask. Here v in 0..k where 0 is the start (S) and 1..k correspond to candy nodes.
    dp = [[INF] * (k + 1) for _ in range(1 << k)]
    dp[0][0] = 0  # start at S with no candy visited.
    
    for mask in range(1 << k):
        for v in range(k + 1):
            cur_cost = dp[mask][v]
            if cur_cost == INF:
                continue
            # Try to go to an unvisited candy j (represented as node j, j=1..k)
            for j in range(1, k + 1):
                if mask & (1 << (j - 1)):
                    continue
                # Only take transition if path exists from node v to candy j.
                if d[v][j] == INF:
                    continue
                next_mask = mask | (1 << (j - 1))
                new_cost = cur_cost + d[v][j]
                if new_cost < dp[next_mask][j]:
                    dp[next_mask][j] = new_cost

    # Now try finishing the path by going from the current node to the goal (node index k+1)
    # and check that total moves is within T. We'll record the maximum count of candy visited.
    ans = -1
    # Check all DP states.
    for mask in range(1 << k):
        for v in range(k + 1):
            if dp[mask][v] == INF or d[v][k+1] == INF:
                continue
            total_moves = dp[mask][v] + d[v][k+1]
            if total_moves <= T:
                candy_count = bin(mask).count("1")
                if candy_count > ans:
                    ans = candy_count
    # Also consider going directly from S to G.
    if d[0][k+1] <= T:
        ans = max(ans, 0)
    
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()