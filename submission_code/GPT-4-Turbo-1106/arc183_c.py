MOD = 998244353

def count_permutations(N, M, conditions):
    dp = [[0] * (N + 2) for _ in range(N + 1)]
    dp[0][0] = 1

    for i in range(1, N + 1):
        s = [0] * (N + 2)
        for j in range(N + 2):
            s[j] = s[j - 1] + dp[i - 1][j]
            s[j] %= MOD
        for j in range(N + 1):
            dp[i][j] = s[j]
            dp[i][j] %= MOD

    for L, R, X in conditions:
        ndp = [[0] * (N + 2) for _ in range(N + 1)]
        for i in range(N + 1):
            for j in range(N + 1):
                if j < L - 1 or j > R:
                    ndp[i][j] = dp[i][j]
                elif L <= j < X:
                    ndp[i][j] = dp[i][j] - dp[i][j - 1]
                elif X < j <= R:
                    ndp[i][j] = dp[i][j] - dp[i - 1][j]
                ndp[i][j] %= MOD
        dp = ndp

    result = sum(dp[N]) % MOD
    return result

def main():
    N, M = map(int, input().split())
    conditions = [tuple(map(int, input().split())) for _ in range(M)]
    print(count_permutations(N, M, conditions))

if __name__ == "__main__":
    main()