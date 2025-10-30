def solve():
    n, m = map(int, input().split())
    mod = 998244353

    fact = [1] * (n * m + 1)
    for i in range(2, n * m + 1):
        fact[i] = (fact[i - 1] * i) % mod

    def power(a, b):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res

    def inv(n):
        return power(n, mod - 2)

    ans = 0
    for k in range(1, n * m + 1):
        term = (fact[k] * fact[n * m] * inv(fact[n * m - k])) % mod
        ans = (ans + term) % mod

    print(ans)

solve()