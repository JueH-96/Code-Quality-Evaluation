def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    
    MOD = 998244353
    
    # Edge case: if K=0, the black ball stays at position 1, answer is 1
    # but K >= 1 per constraints, so we don't strictly need that.
    
    # We use the derived formula:
    # E[K] = (N+1)/2 + ((1 - N)/2)*((N-2)/N)^K  (all operations done mod 998244353)
    
    # We'll define a function for modular inverse:
    def modinv(x, m=MOD):
        return pow(x, m-2, m)  # Fermat's little theorem since m is prime
    
    # Convert N into modulo space
    n_mod = N % MOD
    
    # precompute inverse of 2 mod M
    inv2 = (MOD + 1) // 2  # or modinv(2)
    
    # First term: (N+1)/2 mod
    term1 = (n_mod + 1) % MOD
    term1 = (term1 * inv2) % MOD
    
    # Second term coefficient: (1 - N)/2 mod
    t = (1 - n_mod) % MOD
    term2 = (t * inv2) % MOD
    
    # ratio = (N-2)/N mod
    # (N-2) mod => (n_mod-2) mod
    # we'll invert n_mod in modulo
    num = (n_mod - 2) % MOD
    den = n_mod  # guaranteed not zero modulo, since 1 <= N < MOD
    inv_den = modinv(den)
    ratio = (num * inv_den) % MOD
    
    # ratio^K mod
    ratio_pow_k = pow(ratio, K, MOD)
    
    # Combine:
    # E[K] mod = term1 + term2 * ratio_pow_k mod
    ans = (term1 + term2 * ratio_pow_k) % MOD
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()