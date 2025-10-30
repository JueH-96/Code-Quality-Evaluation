import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    # The number of ways is given by the chromatic polynomial of a cycle graph C_N
    # evaluated at M: P(C_N, M) = (M-1)^N + (-1)^N * (M-1).
    # All calculations are done modulo MOD.

    # M-1. Constraints: M >= 2, so M-1 >= 1.
    val_M_minus_1 = M - 1 
    
    # Calculate term1 = (M-1)^N mod MOD
    # pow(base, exp, mod) computes (base^exp) % mod.
    # Python's pow handles base % mod internally if base >= mod or base < 0.
    # Since M-1 >= 1, this is fine.
    term1 = pow(val_M_minus_1, N, MOD)

    # Calculate term2 = (-1)^N * (M-1) mod MOD
    # First, (M-1) mod MOD
    val_M_minus_1_mod = val_M_minus_1 % MOD
    
    term2: int
    if N % 2 == 1:  # N is odd, so (-1)^N = -1
        # term2 is -(M-1) mod MOD.
        # (MOD - val_M_minus_1_mod) ensures the result is in [0, MOD-1].
        term2 = (MOD - val_M_minus_1_mod) % MOD 
    else:  # N is even, so (-1)^N = 1
        # term2 is (M-1) mod MOD.
        term2 = val_M_minus_1_mod
        
    ans = (term1 + term2) % MOD
    
    print(ans)

if __name__ == '__main__':
    main()