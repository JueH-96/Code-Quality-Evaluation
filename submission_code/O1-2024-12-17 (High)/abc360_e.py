def main():
    import sys
    MOD = 998244353
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    
    # Special case: if N = 1, there's only one ball (black), which stays at position 1 forever.
    if N == 1:
        print(1)
        return
    
    # Modular inverse of 2 in this prime field (known constant for 998244353).
    inv2 = (MOD + 1) // 2  # 499122177
    
    # Compute the modular inverse of N using Fermat's little theorem since MOD is prime.
    invN = pow(N, MOD - 2, MOD)  
    
    # Let alpha = 1 - 2/N (mod MOD).
    alpha = (1 - (2 * invN) % MOD) % MOD

    # From the derived formula, the expected position after K steps is:
    # E[K] = A * (alpha^K) + B, where:
    #   A = (1 - N)/2  (mod MOD)
    #   B = (N + 1)/2  (mod MOD)
    #   alpha = 1 - 2/N
    # We compute everything in modular arithmetic.
    
    A = (1 - N) % MOD
    A = (A * inv2) % MOD  # A = (1 - N)/2 mod
    
    B = (N + 1) % MOD
    B = (B * inv2) % MOD  # B = (N + 1)/2 mod
    
    # Compute alpha^K in O(log K) time.
    p = pow(alpha, K, MOD)
    
    # Final answer = A * (alpha^K) + B (mod MOD).
    ans = (A * p + B) % MOD
    print(ans)

# Do not forget to call main().
if __name__ == "__main__":
    main()