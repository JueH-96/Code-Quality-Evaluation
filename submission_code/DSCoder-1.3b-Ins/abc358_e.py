import sys

def solve():
    K = int(sys.stdin.readline().strip())
    C = list(map(int, sys.stdin.readline().strip().split()))

    mod = 998244353
    dp = [[0] * (K + 1) for _ in range(27)]
    dp[26] = [1] * (K + 1)

    for i in range(26, -1, -1):
        for j in range(K + 1):
            for k in range(C[i] + 1):
                if j >= k:
                    dp[i][j] = (dp[i][j] + dp[i + 1][j - k]) % mod

    ans = 0
    for j in range(K + 1):
        ans = (ans + dp[0][j]) % mod

    print(ans)

solve()