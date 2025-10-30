# YOUR CODE HERE
import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
    
    w = list(map(int, sys.stdin.readline().split()))
    a = list(map(int, sys.stdin.readline().split()))
    
    ans = 0
    
    nodes = sorted(range(n), key=lambda x: w[x])
    
    for node in nodes:
        ans += a[node]
        
        options = []
        for neighbor in adj[node]:
            options.append((w[neighbor], neighbor))
        
        options.sort()
        
        
        remaining = a[node]
        
        for weight, neighbor in options:
            
            transfer = min(remaining, a[neighbor])
            a[neighbor] -= transfer
            remaining -= transfer
            
        
        
        dp = [0] * (w[node] + 1)
        dp[0] = 1
        
        for neighbor in adj[node]:
            new_dp = [0] * (w[node] + 1)
            for i in range(w[node] + 1):
                if dp[i] > 0:
                    for j in range(min(remaining + 1, a[neighbor] + 1)):
                        if i + j * w[neighbor] <= w[node]:
                            new_dp[i + j * w[neighbor]] = (new_dp[i + j * w[neighbor]] + dp[i])
            dp = new_dp
        
        
        for i in range(w[node]):
            if dp[i] > 0:
                transfer = min(remaining, dp[i])
                ans += transfer
                remaining -= transfer

    print(ans)

solve()