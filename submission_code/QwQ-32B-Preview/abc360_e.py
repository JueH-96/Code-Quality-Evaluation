MOD = 998244353

def mod_inverse(a, mod=MOD):
    return pow(a, mod-2, mod)

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_K = sys.stdin.read().split()
    N = int(N_K[0])
    K = int(N_K[1])
    
    if N == 1:
        print(1)
        return
    
    # Compute (1 - N) % MOD
    A = (1 - N) % MOD
    # Compute (N - 2)^K % MOD
    B = pow(N - 2, K, MOD)
    # Compute N^K % MOD
    C = pow(N, K, MOD)
    # Compute 2 * C % MOD and its inverse
    D = (2 * C) % MOD
    inv_D = mod_inverse(D)
    # Compute A * B % MOD
    E = (A * B) % MOD
    # Compute E * inv_D % MOD
    F = (E * inv_D) % MOD
    # Compute (N + 1) * inv(2) % MOD
    G = ((N + 1) * mod_inverse(2)) % MOD
    # Compute F + G % MOD
    H = (F + G) % MOD
    print(H)

if __name__ == "__main__":
    main()