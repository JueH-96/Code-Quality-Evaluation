def power(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return res

def inverse(n, mod):
    return power(n, mod - 2, mod)

def solve():
    n, k = map(int, input().split())
    mod = 998244353
    
    if n == 1:
        print(1)
        return
        
    term1_num = (n + 1) % mod
    term1_den_inv = inverse(2, mod)
    term1 = (term1_num * term1_den_inv) % mod
    
    term2_num = (1 - n) % mod
    if term2_num < 0:
        term2_num += mod
    term2_den_inv = inverse(2, mod)
    term2 = (term2_num * term2_den_inv) % mod
    
    ratio_num = (n - 2) % mod
    if ratio_num < 0:
        ratio_num += mod
    ratio_den_inv = inverse(n, mod)
    ratio = (ratio_num * ratio_den_inv) % mod
    
    ratio_power_k = power(ratio, k, mod)
    
    term3 = (term2 * ratio_power_k) % mod
    
    result = (term1 + term3) % mod
    print(result)

if __name__ == '__main__':
    solve()