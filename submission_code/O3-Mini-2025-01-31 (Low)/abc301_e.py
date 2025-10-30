def main():
    import sys
    from collections import deque
    input = sys.stdin.readline
    INF = 10**9

    H, W, T = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]
    
    # Locate start S, goal G, and candy positions 'o'
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
    
    # Prepare list of important points.
    # node 0: start, node 1..n: candies, node n+1: goal.
    points = [start] + candies + [goal]
    total_points = len(points)
    n_candies = len(candies)
    
    # Pre-calculate pairwise distances among the important points using BFS.
    d = [[INF] * total_points for _ in range(total_points)]
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for idx, (si, sj) in enumerate(points):
        dist = [[-1] * W for _ in range(H)]
        dq = deque()
        dq.append((si, sj))
        dist[si][sj] = 0
        while dq:
            ci, cj = dq.popleft()
            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#' and dist[ni][nj] == -1:
                    dist[ni][nj] = dist[ci][cj] + 1
                    dq.append((ni, nj))
        # Record distances from point idx to every other important point.
        for jdx, (ti, tj) in enumerate(points):
            if dist[ti][tj] != -1:
                d[idx][jdx] = dist[ti][tj]
    
    # Early check: if direct path from start to goal is not possible within T moves then it's impossible.
    if d[0][total_points-1] > T:
        # Even if one detour helps you pick up a candy, total cost is at least the direct cost.
        print(-1)
        return
    
    # DP with state dp[mask][i] represents the minimum moves needed to reach node i having collected
    # candy squares indicated by bitmask mask. Node indices:
    #    0: start, 1..n_candies: candy squares, (n_candies+1): goal.
    # When finishing, we move to goal and do not take any additional moves.
    dp = [[INF] * total_points for _ in range(1 << n_candies)]
    dp[0][0] = 0  # start at S with no candy collected.
    
    for mask in range(1 << n_candies):
        for cur in range(total_points):
            cur_cost = dp[mask][cur]
            if cur_cost == INF:
                continue
            # If we have already reached goal (node total_points-1) this state is terminal.
            if cur == total_points - 1:
                continue
            # Option 1: move to the goal and finish.
            cost_to_goal = d[cur][total_points - 1]
            if cost_to_goal < INF:
                new_cost = cur_cost + cost_to_goal
                if new_cost < dp[mask][total_points - 1]:
                    dp[mask][total_points - 1] = new_cost
            # Option 2: move to a candy that has not been collected yet.
            # Candy nodes are from 1 to n_candies.
            for candy in range(1, n_candies + 1):
                if mask & (1 << (candy - 1)):
                    continue
                cost_to_candy = d[cur][candy]
                if cost_to_candy == INF:
                    continue
                new_cost = cur_cost + cost_to_candy
                new_mask = mask | (1 << (candy - 1))
                if new_cost < dp[new_mask][candy]:
                    dp[new_mask][candy] = new_cost

    ans = -1
    # Check all states which have reached the goal (node total_points-1)
    for mask in range(1 << n_candies):
        if dp[mask][total_points - 1] <= T:
            # count number of candies collected.
            candy_count = bin(mask).count("1")
            ans = max(ans, candy_count)
    print(ans)


if __name__ == "__main__":
    main()