import sys

MOD = 998244353

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    dp = [0] * (1000001)
    dp[0] = 1
    for a in A:
        for i in range(a, 1000001):
            dp[i] = (dp[i] + dp[i-a]) % MOD

    inv = [0] * (1000001)
    inv[1] = 1
    for i in range(2, 1000001):
        inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD

    fact = [0] * (N + 1)
    factinv = [0] * (N + 1)
    fact[0] = factinv[0] = 1
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i % MOD
        factinv[i] = factinv[i-1] * inv[i] % MOD

    ans = 0
    for i in range(N):
        ans = (ans + fact[N-1] * factinv[i] % MOD * factinv[N-1-i] % MOD * dp[10] % MOD) % MOD
    print(ans)

solve()