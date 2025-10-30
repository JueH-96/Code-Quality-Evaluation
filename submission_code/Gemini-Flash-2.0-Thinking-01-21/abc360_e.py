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
    n, k = map(int, input().split())
    mod = 998244353
    if n == 1:
        print(1)
        return
    
    n_inv = power(n, mod - 2, mod)
    c = ((n - 2) * n_inv) % mod
    c_k = power(c, k, mod)
    
    n_plus_1_half = ((n + 1) * power(2, mod - 2, mod)) % mod
    n_minus_1_half = ((n - 1) * power(2, mod - 2, mod)) % mod
    
    term2 = (n_minus_1_half * c_k) % mod
    result = (n_plus_1_half - term2) % mod
    print(result)

if __name__ == '__main__':
    solve()