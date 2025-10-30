import sys, math
def solve():
    import sys
    import sys
    from sys import stdin
    N, M = map(int, stdin.readline().split())
    adj = [[] for _ in range(N)]
    for _ in range(M):
        U, V, W = map(int, stdin.readline().split())
        adj[U-1].append( (V-1, W) )
    INF = float('inf')
    dp = [ [INF] * (1 << N) for _ in range(N) ]
    for node in range(N):
        dp[node][1 << node] = 0
    for mask in range(1 << N):
        for node in range(N):
            if not (mask & (1 << node)):
                continue
            current = dp[node][mask]
            if current == INF:
                continue
            for (next_node, weight) in adj[node]:
                new_mask = mask | (1 << next_node)
                new_weight = current + weight
                if new_weight < dp[next_node][new_mask]:
                    dp[next_node][new_mask] = new_weight
    full_mask = (1 << N) -1
    res = INF
    for node in range(N):
        if dp[node][full_mask] < res:
            res = dp[node][full_mask]
    if res < INF:
        print(int(res))
    else:
        print("No")