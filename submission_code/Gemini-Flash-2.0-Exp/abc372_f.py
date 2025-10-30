def solve():
    n, m, k = map(int, input().split())
    edges = []
    for i in range(m):
        x, y = map(int, input().split())
        edges.append((x, y))

    adj = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        adj[i].append((i % n) + 1)
    for x, y in edges:
        adj[x].append(y)

    dp = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][1] = 1

    for i in range(k):
        for j in range(1, n + 1):
            if dp[i][j] > 0:
                for neighbor in adj[j]:
                    dp[i+1][neighbor] = (dp[i+1][neighbor] + dp[i][j]) % 998244353

    ans = 0
    for i in range(1, n + 1):
        ans = (ans + dp[k][i]) % 998244353

    print(ans)

solve()