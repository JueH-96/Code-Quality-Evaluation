def solve():
    n, m, k = map(int, input().split())
    edges = []
    for i in range(1, n + 1):
        edges.append((i, (i % n) + 1))
    for _ in range(m):
        x, y = map(int, input().split())
        edges.append((x, y))
    
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
    
    dp = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][1] = 1
    
    mod = 998244353
    
    for i in range(k):
        for u in range(1, n + 1):
            if dp[i][u] > 0:
                for v in adj[u]:
                    dp[i+1][v] = (dp[i+1][v] + dp[i][u]) % mod
    
    ans = 0
    for i in range(1, n + 1):
        ans = (ans + dp[k][i]) % mod
    
    print(ans)

solve()