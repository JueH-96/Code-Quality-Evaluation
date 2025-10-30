# YOUR CODE HERE
def solve():
    n, m, k = map(int, input().split())
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append((i + 1) % n)
    for _ in range(m):
        x, y = map(int, input().split())
        adj[x - 1].append(y - 1)

    dp = [[0] * n for _ in range(k + 1)]
    dp[0][0] = 1
    
    for i in range(1, k + 1):
        for j in range(n):
            for neighbor in adj[j]:
                dp[i][neighbor] = (dp[i][neighbor] + dp[i - 1][j]) % 998244353
    
    print(dp[k][0])

solve()