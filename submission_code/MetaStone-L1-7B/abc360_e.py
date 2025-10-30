MOD = 998244353

def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    
    if N == 1:
        print(1)
        return
    
    # Compute a = (N*(N-1) -4)/(N*(N-1)) mod MOD
    numerator_a = (N * (N - 1) - 4) % MOD
    denominator = N * (N - 1) % MOD
    inv_denominator = pow(denominator, MOD-2, MOD)
    a = numerator_a * inv_denominator % MOD
    
    # Compute b = (2*(N+1))/(N*(N-1)) mod MOD
    numerator_b = 2 * (N + 1) % MOD
    b = numerator_b * inv_denominator % MOD
    
    # Compute a^K mod MOD
    a_pow = pow(a, K, MOD)
    
    # Compute term1 = (1 - a^K) * b / (N-1) mod MOD
    numerator_term1 = (1 - a_pow) % MOD
    term1 = numerator_term1 * b % MOD
    term1 = term1 * pow(N - 1, MOD-2, MOD) % MOD
    
    # Compute E = a^K + term1 mod MOD
    E = (a_pow + term1) % MOD
    
    print(E)

if __name__ == "__main__":
    main()