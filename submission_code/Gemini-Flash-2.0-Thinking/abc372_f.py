def solve():
    n, m, k = map(int, input().split())
    adj = [[] for _ in range(n + 1)]
    mod = 998244353

    for i in range(1, n + 1):
        adj[i].append((i % n) + 1)

    for _ in range(m):
        u, v = map(int, input().split())
        adj[u].append(v)

    dp = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][1] = 1

    for moves in range(1, k + 1):
        for current_node in range(1, n + 1):
            if dp[moves - 1][current_node] > 0:
                for neighbor in adj[current_node]:
                    dp[moves][neighbor] = (dp[moves][neighbor] + dp[moves - 1][current_node]) % mod

    ans = 0
    for i in range(1, n + 1):
        ans = (ans + dp[k][i]) % mod
    print(ans)

solve()