def solve():
    n, m, k = map(int, input().split())
    extra_edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        extra_edges.append((u, v))
    
    adj = [[] for _ in range(n + 1)]
    for i in range(1, n):
        adj[i].append(i + 1)
    adj[n].append(1)
    for u, v in extra_edges:
        adj[u].append(v)
        
    dp = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][1] = 1
    
    mod = 998244353
    
    for step in range(1, k + 1):
        for u in range(1, n + 1):
            for v in adj[u]:
                dp[step][v] = (dp[step][v] + dp[step - 1][u]) % mod
                
    result = 0
    for v in range(1, n + 1):
        result = (result + dp[k][v]) % mod
        
    print(result)

if __name__ == '__main__':
    solve()