n, m = map(int, input().split())

MOD = 998244353

if m == 1:
    print(0)
else:
    base = (m - 1) % MOD
    ans = pow(base, n, MOD)
    if n % 2 == 0:
        ans = (ans + base) % MOD
    else:
        ans = (ans - base + MOD) % MOD
    print(ans)