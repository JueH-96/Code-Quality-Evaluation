import sys
# Setting recursion depth limit might not be necessary if KMP implementation is iterative, but can be helpful.
# sys.setrecursionlimit(2000) 

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        """
        Calculates the number of ways to transform string s to string t using exactly k operations.
        An operation consists of removing a suffix of length l (0 < l < n) and prepending it.
        This operation is equivalent to a left cyclic shift by l positions, where l can be 1, ..., n-1.

        Args:
            s: The initial string.
            t: The target string.
            k: The exact number of operations allowed.

        Returns:
            The number of ways modulo 10^9 + 7.
        """
        MOD = 10**9 + 7
        n = len(s)

        # Modular exponentiation function: computes (base^exp) % MOD
        def power(base, exp):
            res = 1
            base %= MOD
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % MOD
                base = (base * base) % MOD
                exp //= 2
            return res

        # Modular inverse function using Fermat's Little Theorem: computes (n_local^(MOD-2)) % MOD
        # This is valid because MOD is prime and n < MOD.
        def modInverse(n_local):
            return power(n_local, MOD - 2)

        # Step 1: Check if t is a cyclic shift of s.
        # Concatenate s with itself. If t is a cyclic shift of s, t must be a substring of s+s.
        s_double = s + s
        # Use string find method. It returns the first index of occurrence or -1 if not found.
        first_occurrence_index = s_double.find(t)

        # If t is not found in s+s, it's impossible to reach t from s via cyclic shifts.
        if first_occurrence_index == -1:
             # Note: If s == t, find returns 0, so this branch is only taken if s != t and t is not a cyclic shift.
             return 0
        
        # If t is found, `first_occurrence_index` (let's call it p) is the required net left shift amount modulo n.
        p = first_occurrence_index
        # Sanity check: if s == t, find returns 0 (p=0). If s != t and t is a cyclic shift, find returns p where 0 < p < n.

        # Step 2: Determine the smallest period `d` of s such that d divides n.
        # This is needed because if s has period d, shifts by p and p+d result in the same string.
        # We use the KMP algorithm's preprocessing step (LPS array computation) to find the period.
        def compute_lps(pattern):
            """ Computes the Longest Proper Prefix which is also Suffix (LPS) array. """
            m = len(pattern)
            lps = [0] * m
            length = 0 # length of the previous longest prefix suffix
            i = 1
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        # Fall back to the LPS length of the previous character in the prefix
                        length = lps[length - 1]
                    else:
                        # No prefix matched suffix ending at i
                        lps[i] = 0
                        i += 1
            return lps

        lps = compute_lps(s)
        # pi[n-1] is the length of the longest proper prefix of s that is also a suffix.
        pi_n_minus_1 = lps[n-1] if n > 0 else 0 
        
        d = n # Assume the period is n by default (no smaller period)
        # The length of the potential smallest repeating unit (border length) is k_period = n - pi[n-1].
        # If k_period divides n, then it's the smallest period `d`.
        if pi_n_minus_1 > 0: # Only if there is a non-empty border
             k_period = n - pi_n_minus_1
             if n % k_period == 0:
                 d = k_period
        # If pi_n_minus_1 is 0, k_period = n. n % n == 0, so d remains n. This is correct.

        # Step 3: Calculate the number of sequences of k operations resulting in a net shift.
        # Let A_k be the number of ways to achieve net shift 0 mod n in k steps.
        # Let X_k be the number of ways to achieve net shift j mod n in k steps, for any j != 0.
        # These can be computed using closed-form formulas derived from matrix exponentiation or recurrence relations.
        
        # Calculate components: (n-1)^k mod MOD and (-1)^k mod MOD
        N1k = power(n - 1, k) 
        Nk = 1 if k % 2 == 0 else (MOD - 1) # (-1)^k is 1 if k is even, -1 (MOD-1) if k is odd
        
        # Calculate modular inverse of n
        n_inv = modInverse(n)
        
        # Compute helper value C = ( (N1k - Nk) * n_inv ) % MOD
        term_in_bracket = (N1k - Nk + MOD) % MOD # Add MOD to ensure the term is non-negative
        C = (term_in_bracket * n_inv) % MOD
        
        # Compute A_k and X_k
        A_k = (Nk + C) % MOD # A_k = ((-1)^k + C) % MOD
        X_k = C              # X_k = C
        
        # Step 4: Combine results based on target shift p and period d.
        # If s == t (p=0), the target shift can be any multiple of d (0, d, 2d, ...).
        # The total number of ways is the sum of ways for each valid shift: A_k + (n/d - 1) * X_k.
        # If s != t (p!=0), the target shifts are p, p+d, p+2d, ... All are non-zero.
        # The total number of ways is the sum of ways for each valid shift: (n/d) * X_k.

        if p == 0: # Case s == t
            # Number of ways = A_k + (number of non-zero shifts that result in s) * X_k
            # Number of non-zero shifts = n/d - 1
            count_factor = n // d - 1
            term2 = (count_factor * X_k) % MOD # Calculate this term modulo MOD
            result = (A_k + term2) % MOD
        else: # Case s != t
            # Number of ways = (number of shifts that result in t) * X_k
            # Number of shifts = n/d
            count_factor = n // d
            result = (count_factor * X_k) % MOD
            
        # The result is already modulo MOD due to calculations.
        # Python's % operator handles potential negative results correctly for positive modulus.
        return result