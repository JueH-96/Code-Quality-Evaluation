# YOUR CODE HERE
MOD = 998244353

def solve():
    N, M = map(int, input().split())
    conditions = [list(map(int, input().split())) for _ in range(M)]

    dp = [[0] * (N + 1) for _ in range(N + 1)]
    dp[0][0] = 1

    for i in range(1, N + 1):
        prefix_sum = [0] * (N + 2)
        for j in range(N + 1):
            prefix_sum[j + 1] = (prefix_sum[j] + dp[i - 1][j]) % MOD

        for j in range(1, N + 1):
            dp[i][j] = prefix_sum[j]
            for L, R, X in conditions:
                if L <= i <= R and X == j:
                    dp[i][j] = (dp[i][j] - (prefix_sum[R] - prefix_sum[L - 1])) % MOD

    print(dp[N][N])

solve()