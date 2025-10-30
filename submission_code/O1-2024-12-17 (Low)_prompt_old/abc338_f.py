def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = input_data[2:]

    # Parse edges
    adj = [[float('inf')] * N for _ in range(N)]
    for i in range(N):
        adj[i][i] = 0  # distance to self is 0

    idx = 0
    for _ in range(M):
        u = int(edges[idx]); v = int(edges[idx+1]); w = int(edges[idx+2])
        idx += 3
        u, v = u-1, v-1
        adj[u][v] = w

    # Floyd-Warshall to get all-pairs shortest path
    for k in range(N):
        for i in range(N):
            if adj[i][k] == float('inf'):
                continue
            for j in range(N):
                if adj[k][j] == float('inf'):
                    continue
                new_dist = adj[i][k] + adj[k][j]
                if new_dist < adj[i][j]:
                    adj[i][j] = new_dist

    # Check connectivity: if there's a vertex i such that
    # no path from any j is possible to i or from i to any j, 
    # it doesn't necessarily mean "No" here, because we only need
    # to see if there's *some* walk that covers all vertices.
    # We'll rely on DP to see if we can form such a route.

    # DP (bitmask) approach:
    # dp[mask][i] = minimum cost to have visited the set of vertices "mask" (in binary),
    #               ending in vertex i
    # We want min(dp[(1<<N)-1][i]) over i.

    all_visited = (1 << N) - 1
    dp = [[float('inf')] * N for _ in range(1 << N)]

    # Initialize dp for starting on each vertex
    for i in range(N):
        dp[1 << i][i] = 0

    for mask in range(1 << N):
        for i in range(N):
            if dp[mask][i] == float('inf'):
                continue
            # try to go to a next vertex j
            for j in range(N):
                if adj[i][j] == float('inf'):
                    continue
                next_mask = mask | (1 << j)
                cost = dp[mask][i] + adj[i][j]
                if cost < dp[next_mask][j]:
                    dp[next_mask][j] = cost

    ans = min(dp[all_visited][i] for i in range(N))

    if ans == float('inf'):
        print("No")
    else:
        print(ans)

def __starting_point():
    solve()

if __name__ == "__main__":
    solve()