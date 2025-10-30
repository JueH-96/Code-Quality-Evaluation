import sys
# Optional: Increase recursion depth if needed, though unlikely for this problem
# sys.setrecursionlimit(2000) 

# Global precomputation for factorials and inverse factorials
MOD = 10**9 + 7
# Maximum possible value of N = m*n is 10^5. We need factorials up to N-2.
# Precomputing up to 10^5 is sufficient. MAX_N_needed = 100000.
MAX_N_needed = 100000 
# The size of the arrays should be MAX_N_needed + 1 to store indices from 0 up to MAX_N_needed
MAX_N_SIZE = MAX_N_needed + 1 

# Precompute factorials modulo MOD
fact = [1] * MAX_N_SIZE
for i in range(1, MAX_N_SIZE):
    fact[i] = (fact[i-1] * i) % MOD

# Precompute inverse factorials modulo MOD
invfact = [1] * MAX_N_SIZE
# Calculate inverse factorial of MAX_N_SIZE-1 using Fermat's Little Theorem: a^(p-2) = a^-1 mod p
# pow(base, exponent, modulus) computes (base^exponent) % modulus efficiently
invfact[MAX_N_SIZE-1] = pow(fact[MAX_N_SIZE-1], MOD - 2, MOD)

# Calculate inverse factorials downwards using the relation invfact[i] = invfact[i+1] * (i+1) % MOD
for i in range(MAX_N_SIZE-2, -1, -1):
    invfact[i] = (invfact[i+1] * (i+1)) % MOD

def combinations_util(n, k, mod):
    """ Helper function to calculate combinations C(n, k) modulo mod using precomputed factorials """
    # Basic validity checks for k
    if k < 0 or k > n:
        return 0
    # Base cases for combinations
    if k == 0 or k == n:
        return 1
    # Optimization: C(n, k) == C(n, n-k). Use the smaller value between k and n-k.
    if k > n // 2: 
        k = n - k
    
    # Ensure n is within the precomputed range. N = m*n <= 10^5, so n can be up to 10^5.
    # We calculate C(N-2, k-2). Max value for n here is N-2, which is at most 10^5 - 2.
    # This is within our precomputed range [0, 100000].
    if n >= MAX_N_SIZE:
         # This check is mostly defensive, based on constraints it should not be triggered.
         raise ValueError(f"N={n} is too large for precomputed factorials up to {MAX_N_SIZE-1}")

    # Calculate C(n, k) = n! / (k! * (n-k)!) mod p
    # This is equivalent to = n! * (k!)^{-1} * ((n-k)!)^(-1) % mod
    # Use precomputed factorials (fact) and inverse factorials (invfact)
    numerator = fact[n]
    # The inverse of the denominator is (k!)^{-1} * ((n-k)!)^(-1) mod mod
    denominator_inv = (invfact[k] * invfact[n-k]) % mod
    # Final result is (numerator * denominator_inv) % mod
    return (numerator * denominator_inv) % mod

class Solution:
    """ 
    Implements the solution to calculate the total sum of Manhattan distances 
    between all pairs of pieces, summed over all valid arrangements.
    """
    def distanceSum(self, m: int, n: int, k: int) -> int:
        """
        Calculates the sum of Manhattan distances between all pairs of pieces,
        summed over all valid arrangements of k pieces on an m x n grid.
        The result is returned modulo 10^9 + 7.

        Args:
            m: The number of rows in the grid.
            n: The number of columns in the grid.
            k: The number of identical pieces to place on the grid.

        Returns:
            The total sum of Manhattan distances modulo 10^9 + 7.
        """
        
        N = m * n # Total number of cells in the grid

        # Use the globally defined MOD constant
        # MOD = 10**9 + 7 # It's defined globally

        # Check constraints based on the problem statement guarantees:
        # 2 <= k <= N
        # N = m*n >= 2
        # If these were not guaranteed, we'd need checks:
        if k < 2: return 0 # No pairs of pieces if k < 2, sum is 0.
        if N < k: return 0 # Impossible to place k pieces if N < k.

        # Calculate Modular Inverse of 6.
        # Needed for dividing by 6 in the Sr and Sc formulas.
        # Since MOD = 10^9 + 7 is a prime number greater than 3, 
        # 6 has a modular multiplicative inverse modulo MOD.
        inv6 = pow(6, MOD - 2, MOD)

        # Calculate Sr: Sum of row distance contributions over all pairs of distinct cells.
        # Formula derived as Sr = (n^2 * m * (m^2-1) / 6) % MOD
        # We use Python's arbitrary precision integers and apply modulo at intermediate steps.
        term_n_sq = (n * n) % MOD  
        term_m = m % MOD
        m_sq = (m * m) % MOD
        # Calculate (m^2 - 1) mod MOD carefully. Add MOD before taking modulo to ensure non-negativity.
        term_m_sq_minus_1 = (m_sq - 1 + MOD) % MOD 
        
        Sr_numerator = (term_n_sq * term_m) % MOD
        Sr_numerator = (Sr_numerator * term_m_sq_minus_1) % MOD
        Sr = (Sr_numerator * inv6) % MOD

        # Calculate Sc: Sum of column distance contributions over all pairs of distinct cells.
        # Formula derived as Sc = (m^2 * n * (n^2-1) / 6) % MOD
        term_m_sq = (m * m) % MOD
        term_n = n % MOD
        n_sq = (n * n) % MOD
        # Calculate (n^2 - 1) mod MOD carefully.
        term_n_sq_minus_1 = (n_sq - 1 + MOD) % MOD
        
        Sc_numerator = (term_m_sq * term_n) % MOD
        Sc_numerator = (Sc_numerator * term_n_sq_minus_1) % MOD
        Sc = (Sc_numerator * inv6) % MOD

        # S_total is the sum of Manhattan distances over all pairs of distinct cells.
        S_total = (Sr + Sc) % MOD

        # The final answer is S_total multiplied by the number of ways any specific pair of cells
        # can be part of an arrangement of k pieces. This count is C(N-2, k-2).
        # Calculate combinations coefficient C(N-2, k-2) using the global helper function.
        # Need to compute C(N-2, k-2). Constraints ensure arguments N-2 and k-2 are valid.
        # N >= 2 and k >= 2 implies N-2 >= 0 and k-2 >= 0.
        # N >= k implies N-2 >= k-2.
        
        # Handles edge case where N-2 < k-2 which would imply N < k. This shouldn't happen
        # with problem constraints but included defensively. If N-2 < k-2, C(N-2, k-2) = 0.
        if N - 2 < k - 2:
           comb_coeff = 0
        else:
           comb_coeff = combinations_util(N - 2, k - 2, MOD)

        # Final answer is the product modulo MOD
        ans = (comb_coeff * S_total) % MOD
        return ans