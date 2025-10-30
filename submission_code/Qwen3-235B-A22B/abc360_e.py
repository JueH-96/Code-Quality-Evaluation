MOD = 998244353

def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    n = N % MOD
    inv_N = pow(n, MOD-2, MOD)
    inv_N2 = inv_N * inv_N % MOD
    
    n_sq = (n * n) % MOD
    
    a = (n_sq - 2 * n + 2) % MOD
    a = a * inv_N2 % MOD
    
    b = (2 * (n - 1)) % MOD
    b = b * inv_N2 % MOD
    
    c = (2 * inv_N2) % MOD
    
    d = (n_sq - 2) % MOD
    d = d * inv_N2 % MOD
    
    p = 1
    q = 0
    
    for _ in range(K):
        new_p = (a * p + b * q) % MOD
        new_q = (c * p + d * q) % MOD
        p, q = new_p, new_q
    
    inv_2 = pow(2, MOD-2, MOD)
    term = (n * p) % MOD
    numerator = ( (n + 2) % MOD - term ) % MOD
    numerator = (numerator + MOD) % MOD
    E = numerator * inv_2 % MOD
    print(E)

if __name__ == "__main__":
    main()