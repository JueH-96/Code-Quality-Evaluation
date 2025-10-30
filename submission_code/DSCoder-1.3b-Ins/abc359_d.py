import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    mod = 998244353
    dp = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 1

    for i in range(1, N + 1):
        for j in range(K + 1):
            if S[i - 1] == '?':
                dp[j][i] = (dp[j - 1][i - 1] + dp[j][i - 1]) % mod
            else:
                dp[j][i] = dp[j][i - 1]
                if j > 0:
                    dp[j][i] += dp[j - 1][i - 1] if S[i - 1] == S[i - 2] else 0
                    dp[j][i] %= mod

    print(dp[K][N])

if __name__ == "__main__":
    solve()