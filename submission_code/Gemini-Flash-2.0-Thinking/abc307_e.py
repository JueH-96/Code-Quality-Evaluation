def power(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        exp //= 2
    return res

def solve():
    n, m = map(int, input().split())
    mod = 998244353

    if m == 1:
        print(0)
        return

    term1 = power(m - 1, n, mod)
    term2 = (m - 1) % mod

    if n % 2 == 0:
        term2_signed = term2
    else:
        term2_signed = (-term2 + mod) % mod

    result = (term1 + term2_signed) % mod
    print(result)

solve()