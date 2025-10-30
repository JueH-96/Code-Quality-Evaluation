# YOUR CODE HERE
from collections import defaultdict

def solve():
    mod = 998244353
    n, m, k = map(int, input().split())
    edges = defaultdict(list)
    for _ in range(m):
        x, y = map(int, input().split())
        edges[x-1].append(y-1)
    
    dp = [[0] * n for _ in range(k+1)]
    dp[0][0] = 1
    
    for i in range(k):
        for j in range(n):
            if j == 0:
                dp[i+1][j] = (dp[i+1][j] + dp[i][j] + dp[i][n-1]) % mod
            else:
                dp[i+1][j] = (dp[i+1][j] + dp[i][j-1] + dp[i][j] + dp[i][j+1]) % mod
            for next_v in edges[j]:
                dp[i+1][next_v] = (dp[i+1][next_v] + dp[i][j]) % mod
    
    print(dp[k][0])

solve()