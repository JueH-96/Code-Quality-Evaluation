def solve():
    import sys
    MOD = 998244353
    
    # Read N as a string to handle large input up to 10^18
    N_str = sys.stdin.readline().strip()
    N_int = int(N_str)
    
    # Number of digits in N
    d = len(N_str)
    
    # Precompute N modulo MOD
    N_mod = N_int % MOD
    
    # r = 10^d (mod MOD)
    r = pow(10, d, MOD)
    
    # If r == 1 mod MOD, the sum of r^k from k=0..N-1 is simply N, so V_N mod MOD = N*N mod MOD
    if r == 1:
        ans = (N_mod * N_mod) % MOD
    else:
        # Use the formula:
        # V_N = N * ( (r^N - 1) / (r - 1) )  (mod MOD)
        numerator = pow(r, N_int, MOD) - 1
        if numerator < 0:
            numerator += MOD
        
        denominator = (r - 1) % MOD
        denom_inv = pow(denominator, MOD - 2, MOD)  # Modular inverse using Fermat's little theorem
        
        summation = (numerator * denom_inv) % MOD
        ans = (N_mod * summation) % MOD
    
    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()