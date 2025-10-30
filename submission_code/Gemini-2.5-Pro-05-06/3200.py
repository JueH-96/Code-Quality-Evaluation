class Solution:
  def stringCount(self, n: int) -> int:
    MOD = 10**9 + 7

    # The problem states n >= 1.
    # The formula derived works for n=1,2,3 giving 0, which is correct as "leet" requires 4 characters.
    # So, an explicit check `if n < 4: return 0` is not strictly needed.

    # Total strings of length n using 26 letters
    total_strings = pow(26, n, MOD)

    # PIE terms calculation:
    # sum1 = 3 * 25^n + n * 25^(n-1)
    term_25_n = pow(25, n, MOD)
    # For n=0, n-1 = -1. pow(25, -1, MOD) would be modular inverse.
    # However, constraint is n >= 1, so n-1 >= 0. pow(25,0,MOD) = 1.
    term_25_n_minus_1 = pow(25, n - 1, MOD) 
    
    # In Python, (A*B + C*D) % MOD correctly handles intermediate values A*B or C*D being larger than MOD.
    # n_val = n % MOD, but since n <= 10^5 < MOD, n_val = n.
    # (3 * term_25_n) can be > MOD. (n * term_25_n_minus_1) can be > MOD.
    # Their sum can be > MOD. Python handles these intermediate large values.
    # One final % MOD is applied to the sum.
    sum1 = (3 * term_25_n + n * term_25_n_minus_1) % MOD
    
    # sum2 = 3 * 24^n + 2n * 24^(n-1)
    term_24_n = pow(24, n, MOD)
    term_24_n_minus_1 = pow(24, n - 1, MOD)
    # (2 * n) is at most 2*10^5, which is less than MOD.
    sum2 = (3 * term_24_n + 2 * n * term_24_n_minus_1) % MOD

    # sum3 = 23^n + n * 23^(n-1)
    term_23_n = pow(23, n, MOD)
    term_23_n_minus_1 = pow(23, n - 1, MOD)
    sum3 = (term_23_n + n * term_23_n_minus_1) % MOD
    
    # Number of "bad" strings (those not satisfying the conditions)
    # bad_strings = Sum1 - Sum2 + Sum3
    # (A - B + MOD) % MOD ensures intermediate result of subtraction is positive.
    bad_strings = (sum1 - sum2 + MOD) % MOD 
    bad_strings = (bad_strings + sum3) % MOD
    
    # Total good strings = Total strings - Bad strings
    ans = (total_strings - bad_strings + MOD) % MOD
    
    return ans