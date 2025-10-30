def solve():
    n, m = map(int, input().split())
    mod = 998244353

    if m == 1:
        if n > 1:
            print(0)
        else:
            print(1)
        return
    
    if n == 2:
        print((m * (m - 1)) % mod)
        return

    ans = (m * pow(m - 1, n - 1, mod)) % mod
    
    if n % 2 == 0:
        ans = (ans - (m * pow(m - 2, n // 2 - 1, mod) * (m - 1) % mod) % mod + mod) % mod
    
    print(ans)

solve()