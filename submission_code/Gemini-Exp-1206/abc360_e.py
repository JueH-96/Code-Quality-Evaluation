def solve():
    n, k = map(int, input().split())
    mod = 998244353

    def modinv(x, mod):
        return pow(x, mod - 2, mod)

    dp = [0] * (n + 1)
    dp[1] = 1

    for _ in range(k):
        new_dp = [0] * (n + 1)
        for i in range(1, n + 1):
            new_dp[i] = (dp[i] * (n * (n - 1) + n) + (sum(dp) - dp[i]) * n) * modinv(n * n, mod) % mod
        dp = new_dp

    ans = 0
    for i in range(1, n + 1):
        ans = (ans + i * dp[i]) % mod
    print(ans)

solve()