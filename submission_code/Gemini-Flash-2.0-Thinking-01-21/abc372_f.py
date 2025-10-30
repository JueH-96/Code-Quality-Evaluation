def solve():
    n, m, k = map(int, input().split())
    edges_m = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges_m.append((u, v))
    
    adj = [[] for _ in range(n + 1)]
    for i in range(1, n):
        adj[i].append(i + 1)
    adj[n].append(1)
    for u, v in edges_m:
        adj[u].append(v)
        
    dp = [0] * (n + 1)
    dp[1] = 1
    
    mod = 998244353
    
    for _ in range(k):
        next_dp = [0] * (n + 1)
        for u in range(1, n + 1):
            if dp[u] > 0:
                for v in adj[u]:
                    next_dp[v] = (next_dp[v] + dp[u]) % mod
        dp = next_dp
        
    result = sum(dp[1:]) % mod
    print(result)

if __name__ == '__main__':
    solve()