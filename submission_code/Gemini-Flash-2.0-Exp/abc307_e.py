def solve():
    n, m = map(int, input().split())
    mod = 998244353

    if n == 2:
        print(m * (m - 1) % mod)
        return

    if m == 1:
        if n > 1:
            print(0)
        else:
            print(1)
        return

    if m == 2:
        if n % 2 == 0:
            print(2)
        else:
            print(0)
        return

    def power(base, exp, mod):
        res = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % mod
            base = (base * base) % mod
            exp //= 2
        return res

    ans = (power(m - 2, n, mod) * (m % mod) % mod + power(-1, n, mod) * (m % mod) % mod) % mod
    print(ans)

solve()