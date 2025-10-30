def solve():
    n, m = map(int, input().split())
    mod = 998244353

    inv2 = pow(2, mod - 2, mod)

    dp = [0] * (n * m + 1)

    for i in range(n * m, 0, -1):
        if i <= n:
            dp[i] = 2
        else:
            dp[i] = (2 + dp[i - n]) * inv2 % mod
    
    ans = 0
    for i in range(1, n * m + 1):
        ans = (ans + dp[i]) % mod
    
    print(ans)

solve()