def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    MOD = 998244353
    
    # Read N as a string
    N_str = sys.stdin.readline().strip()
    N = int(N_str)  # up to 10^18 fits in Python int
    
    # Length of N in decimal
    L = len(N_str)
    
    # Compute 10^L mod MOD
    T = pow(10, L, MOD)
    
    # We will compute the sum of the geometric series: 1 + T + T^2 + ... + T^(N-1) mod MOD
    # sum = (T^N - 1) * inverse(T - 1) mod MOD, if T != 1 mod MOD
    # if T == 1 mod MOD, sum = N mod MOD
    if T == 1:
        sum_power = N % MOD
    else:
        # T^N mod MOD
        tpow = pow(T, N, MOD)
        numerator = (tpow - 1) % MOD
        denominator = (T - 1) % MOD
        # inverse of denominator mod MOD
        inv_denom = pow(denominator, MOD - 2, MOD)
        sum_power = (numerator * inv_denom) % MOD
    
    # N mod MOD
    N_mod = N % MOD
    
    # Final result = N * sum_{k=0}^{N-1} T^k mod MOD
    ans = (N_mod * sum_power) % MOD
    
    # Print the result
    print(ans)

# Do not forget to call main
if __name__ == "__main__":
    main()