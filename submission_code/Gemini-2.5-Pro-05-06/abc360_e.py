MOD = 998244353

def power(a, b):
    # computes (a^b) % MOD
    # Handles a=0, b=0 case as 1 by convention (0^0=1) if b becomes 0 in loop
    # but initial b=0 returns 1 immediately.
    # If a=0, b>0, returns 0.
    # If a is negative, a % MOD makes it positive.
    res = 1
    a %= MOD 
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

def inv(n):
    # computes modular inverse of n modulo MOD
    # n^(MOD-2) % MOD
    return power(n, MOD - 2)

def solve():
    N, K = map(int, input().split())

    if N == 1:
        # Black ball is always at position 1. Expected value is 1.
        print(1)
        return

    # Calculate A_pow_K = ((N-2)/N)^K
    # A_pow_K = (N-2)^K * (N^K)^{-1}
    
    # (N-2)^K
    term_N_minus_2_pow_K = power(N - 2, K)
    
    # (N^K)^{-1}
    term_N_pow_K = power(N, K)
    term_N_pow_K_inv = inv(term_N_pow_K)

    A_pow_K = (term_N_minus_2_pow_K * term_N_pow_K_inv) % MOD

    # Calculate factor (1-N)/2
    inv2 = inv(2)
    term_1_minus_N = (1 - N + MOD) % MOD 
    factor1 = (term_1_minus_N * inv2) % MOD
    
    # First part of E_K formula: A_pow_K * (1-N)/2
    part1 = (A_pow_K * factor1) % MOD
    
    # Calculate factor (N+1)/2
    term_N_plus_1 = (N + 1) % MOD # N+1 can't be negative
    factor2 = (term_N_plus_1 * inv2) % MOD
    
    # Second part of E_K formula is (N+1)/2, which is factor2
    part2 = factor2

    ans = (part1 + part2) % MOD
    
    print(ans)

solve()