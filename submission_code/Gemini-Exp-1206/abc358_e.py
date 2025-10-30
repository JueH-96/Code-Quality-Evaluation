def solve():
    k = int(input())
    c = list(map(int, input().split()))
    mod = 998244353

    dp = [[0] * (k + 1) for _ in range(27)]
    dp[0][0] = 1

    for i in range(26):
        for j in range(k + 1):
            for l in range(min(k - j, c[i]) + 1):
                dp[i + 1][j + l] = (dp[i + 1][j + l] + dp[i][j]) % mod

    ans = 0
    for i in range(1, k + 1):
        ans = (ans + dp[26][i]) % mod

    print(ans)

solve()