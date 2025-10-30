def solve():
    n_str = input()
    n = int(n_str)
    mod = 998244353

    s_n = str(n)
    d = len(s_n)

    p = pow(10, d, mod)

    if p == 1:
        result = (n * n) % mod
    else:
        p_n = pow(p, n, mod)
        numerator = (p_n - 1 + mod) % mod
        denominator = (p - 1 + mod) % mod
        inv_denominator = pow(denominator, mod - 2, mod)
        geometric_sum = (numerator * inv_denominator) % mod
        result = (n * geometric_sum) % mod

    print(result)

solve()