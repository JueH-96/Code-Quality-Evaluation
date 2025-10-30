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
    n_str = input()
    n = int(n_str)
    mod = 998244353
    d = len(n_str)
    r = power(10, d, mod)
    if r == 1:
        print(power(n, 2, mod))
    else:
        r_n = power(r, n, mod)
        u = (r - 1) % mod
        u_inv = inverse(u, mod)
        v_n_mod_m = (n % mod) * ((r_n - 1 + mod) % mod) * u_inv
        print(v_n_mod_m % mod)

if __name__ == '__main__':
    solve()