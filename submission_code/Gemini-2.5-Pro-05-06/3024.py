class Solution:
  def numberOfWays(self, s: str, t: str, k: int) -> int:
    MOD = 10**9 + 7
    n = len(s)

    # Step 1: Check if t is a cyclic shift of s.
    # s.length == t.length is given. n >= 2 is given.
    # If t is not a substring of (s+s), it cannot be a cyclic shift.
    if t not in (s + s):
        return 0
    
    # Step 2: Compute N_D, the number of distinct cyclic shifts of s.
    # This is p_s, the length of the smallest string u such that s = u^p.
    # KMP prefix function pi for s. 
    # pi[i] = length of longest proper prefix of s[0...i] that is also suffix of s[0...i].
    pi = [0] * n
    current_match_len = 0 # Represents length of current matching prefix that's also a suffix
    i = 1 # Start comparing s[1] with s[current_match_len]
    while i < n:
        if s[i] == s[current_match_len]:
            current_match_len += 1
            pi[i] = current_match_len
            i += 1
        else:
            if current_match_len != 0:
                current_match_len = pi[current_match_len - 1]
            else:
                # No shorter prefix/suffix match found, pi[i] is 0.
                pi[i] = 0
                i += 1
    
    # L is length of longest proper prefix of s that is also a suffix of s.
    L = pi[n-1] 
    
    # If n is divisible by (n-L), then the smallest period length is (n-L). This is N_D.
    # Otherwise, s is "primitive" in terms of repetition for cyclic shifts, so N_D = n.
    # (n-L) must be > 0 because L is length of PROPER prefix, so L < n. Thus n-L >= 1.
    if n % (n - L) == 0:
        N_D = n - L
    else:
        N_D = n

    # Step 3 & 4: Eigenvalue based calculation
    
    # Term (n-1)^k mod MOD. Note: n >= 2, so n-1 >= 1.
    term_n_minus_1_pow_k = pow(n - 1, k, MOD)
    
    # Term (-1)^k mod MOD.
    term_minus_1_pow_k = 1 if k % 2 == 0 else (MOD - 1)

    # N_D_inv = N_D^{-1} mod MOD. 
    # N_D >= 1 (since N_D is a length). N_D <= n <= 5*10^5 < MOD. So N_D_inv exists.
    N_D_inv = pow(N_D, MOD - 2, MOD)

    if s == t:
        # Number of ways to get s = ((n-1)^k + (N_D-1)*(-1)^k) / N_D
        # This formula is also correct for N_D=1, where it becomes (n-1)^k.
        
        # (N_D-1) can be 0 if N_D=1. (N_D-1+MOD)%MOD ensures it's non-negative.
        term_N_D_minus_1 = (N_D - 1 + MOD) % MOD 
        
        # ((N_D-1) * (-1)^k) % MOD
        term_combined_N_D_minus_1_and_minus_1_pow_k = (term_N_D_minus_1 * term_minus_1_pow_k) % MOD
        
        numerator = (term_n_minus_1_pow_k + term_combined_N_D_minus_1_and_minus_1_pow_k) % MOD
        ans = (numerator * N_D_inv) % MOD
        return ans
    else: # s != t
        # If N_D = 1:
        #   s is the only distinct cyclic shift. Since s != t, t cannot be this shift.
        #   This case should have been caught by `if t not in (s+s): return 0`.
        #   Explicitly handling it here for safety/clarity.
        if N_D == 1:
             return 0 

        # Number of ways to get t where t != s = ((n-1)^k - (-1)^k) / N_D
        # This formula requires N_D >= 2 (as per derivation involving division by N_D-1 for individual counts).
        
        # ( (n-1)^k - (-1)^k ) % MOD. Add MOD before taking % to handle potential negative result.
        numerator = (term_n_minus_1_pow_k - term_minus_1_pow_k + MOD) % MOD 
        ans = (numerator * N_D_inv) % MOD
        return ans