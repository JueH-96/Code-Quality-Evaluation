import sys
input = sys.stdin.read

def solve():
    MOD = 998244353

    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    dp = [0] * (N + 1)
    dp[0] = 1

    for i in range(1, N + 1):
        dp[i] = (2 * dp[i - 1]) % MOD
        if A[i - 1] != 0:
            dp[i] = (dp[i] - dp[A[i - 1] - 1]) % MOD

    result = dp[N]
    if result < 0:
        result += MOD

    print(result)

solve()