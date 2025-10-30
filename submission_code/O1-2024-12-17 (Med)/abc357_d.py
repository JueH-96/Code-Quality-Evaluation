def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353
    
    # Read N
    N_str = sys.stdin.readline().strip()
    N = int(N_str)
    
    # Number of digits in N
    d = len(N_str)
    
    # Fast exponentiation (base^exp mod m)
    def mod_pow(base, exp, m):
        result = 1
        cur = base % m
        e = exp
        while e > 0:
            if e & 1:
                result = (result * cur) % m
            cur = (cur * cur) % m
            e >>= 1
        return result
    
    # Compute 10^d mod
    r = mod_pow(10, d, MOD)
    N_mod = N % MOD
    
    # If r = 1 mod, the geometric series sumS = N (each term is 1 mod)
    if r == 1:
        sumS = N_mod
    else:
        # sum of 1 + r + r^2 + ... + r^(N-1) mod
        # = (1 - r^N) / (1 - r) mod
        rN = mod_pow(r, N, MOD)
        numerator = (1 - rN) % MOD
        denominator = (1 - r) % MOD
        # Modular inverse via Fermat's little theorem (since MOD is prime)
        inv_denom = mod_pow(denominator, MOD-2, MOD)
        sumS = (numerator * inv_denom) % MOD
    
    # V_N mod = N * sumS mod
    ans = (N_mod * sumS) % MOD
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()