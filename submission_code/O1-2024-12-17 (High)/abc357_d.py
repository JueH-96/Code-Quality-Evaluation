# YOUR CODE HERE
def main():
    import sys
    MOD = 998244353

    # Read N as a string
    N_str = sys.stdin.readline().strip()
    # Convert N to integer
    N_val = int(N_str)
    # Number of digits in N
    L = len(N_str)

    # N modulo MOD
    n_mod = N_val % MOD

    # Compute 10^L modulo MOD
    a = pow(10, L, MOD)

    # If a == 1 mod MOD, then each 10^(k*L) ≡ 1 (mod MOD),
    # summation of N terms is N, so the sum mod MOD is N_mod.
    if a == 1:
        S = n_mod
    else:
        # a^N_val mod MOD
        aN = pow(a, N_val, MOD)
        # Numerator of geometric series sum (a^N - 1) mod MOD
        numerator = (aN - 1) % MOD
        # Denominator (a - 1) mod MOD, then invert
        denom = (a - 1) % MOD
        inv_denom = pow(denom, MOD - 2, MOD)  # Modular inverse
        # Sum of the geometric series mod MOD
        S = (numerator * inv_denom) % MOD

    # The answer is N_mod * S mod MOD
    ans = (n_mod * S) % MOD
    print(ans)

# Do not forget to call main()
main()