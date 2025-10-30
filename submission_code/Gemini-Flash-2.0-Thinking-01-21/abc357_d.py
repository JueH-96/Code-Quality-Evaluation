def power(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return res

def inverse(a, mod):
    return power(a, mod - 2, mod)

def solve():
    n = int(input())
    mod = 998244353
    n_str = str(n)
    d = len(n_str)
    x = power(10, d, mod)
    if x == 1:
        result = (n * n) % mod
    else:
        num = (power(x, n, mod) - 1 + mod) % mod
        den = (x - 1 + mod) % mod
        den_inv = inverse(den, mod)
        result = (n * num * den_inv) % mod
    print(result)

if __name__ == '__main__':
    solve()