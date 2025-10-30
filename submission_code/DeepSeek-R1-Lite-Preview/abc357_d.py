def main():
    MOD = 998244353
    N_str = input().strip()
    N = int(N_str)
    d = len(N_str)
    
    # Compute a = 10^d mod MOD
    a = pow(10, d, MOD)
    
    # Compute a^N mod MOD
    a_power_N = pow(a, N, MOD)
    
    # Compute S = (a^N - 1) * inverse(a - 1) mod MOD
    numerator = (a_power_N - 1) % MOD
    denominator = (a - 1) % MOD
    inverse_denominator = pow(denominator, MOD - 2, MOD)
    S = (numerator * inverse_denominator) % MOD
    
    # Compute V_N = N * S mod MOD
    N_mod = N % MOD
    V_N = (N_mod * S) % MOD
    
    print(V_N)

if __name__ == "__main__":
    main()