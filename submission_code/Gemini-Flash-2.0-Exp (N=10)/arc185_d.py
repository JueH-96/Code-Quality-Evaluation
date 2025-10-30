def solve():
    n, m = map(int, input().split())
    mod = 998244353

    def mod_pow(base, exp, m):
        res = 1
        base %= m
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % m
            base = (base * base) % m
            exp //= 2
        return res

    def mod_inv(a, m):
        return mod_pow(a, m - 2, m)

    ans = 0
    for i in range(1, n * m + 1):
        ans = (ans + (n * mod_inv(min(i, n), mod)) % mod) % mod
    print(ans)

solve()