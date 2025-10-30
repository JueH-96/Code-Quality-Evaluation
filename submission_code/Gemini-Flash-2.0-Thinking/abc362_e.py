def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    results = [0] * n

    dp = [[{} for _ in range(n + 1)] for _ in range(n)]

    for i in range(n):
        dp[i][1][0] = 1

    for k in range(2, n + 1):
        for i in range(n):
            for j in range(i):
                diff = a[i] - a[j]
                if k == 2:
                    dp[i][k][diff] = dp[i][k].get(diff, 0) + 1
                elif k > 2 and diff in dp[j][k - 1]:
                    dp[i][k][diff] = (dp[i][k].get(diff, 0) + dp[j][k - 1][diff]) % mod

    for k_idx in range(n):
        k = k_idx + 1
        count_k = 0
        for i in range(n):
            for diff in dp[i][k]:
                count_k = (count_k + dp[i][k][diff]) % mod
        results[k_idx] = count_k

    print(*results)

solve()