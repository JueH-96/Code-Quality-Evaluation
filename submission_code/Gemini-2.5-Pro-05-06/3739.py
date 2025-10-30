class Solution:
  MOD = 10**9 + 7

  def distanceSum(self, m: int, n: int, k: int) -> int:
    N = m * n

    # Constraints: 2 <= k <= m*n implies k >= 2.
    # m*n >= 2 implies N >= 2.
    # These ensure N-2 >= 0 and k-2 >= 0.
    # Also, N >= k implies N-2 >= k-2 (so C(N-2, k-2) is well-defined).

    # Determine the maximum value X for which X! is needed for C(X, R).
    # We need C(m+1, 3), C(n+1, 3), and C(N-2, k-2).
    # So, X_max is max(m+1, n+1, N-2).
    # Smallest N is 2 (e.g. m=1, n=2). Then N-2=0.
    # m, n >= 1, so m+1, n+1 >= 2. Thus X_max is at least 2.
    X_max = 0
    if m + 1 > X_max: X_max = m + 1
    if n + 1 > X_max: X_max = n + 1
    if N - 2 > X_max: X_max = N - 2 
    
    fact = [1] * (X_max + 1)
    inv_fact = [1] * (X_max + 1)
    
    # Precompute factorials: fact[i] = i! mod MOD
    # fact[0] is already 1. Loop starts from i=1.
    for i in range(1, X_max + 1):
        fact[i] = (fact[i-1] * i) % self.MOD
    
    # Precompute inverse factorials: inv_fact[i] = (i!)^(-1) mod MOD
    # inv_fact[X_max] = (X_max!)^(MOD-2) mod MOD (by Fermat's Little Theorem)
    inv_fact[X_max] = pow(fact[X_max], self.MOD - 2, self.MOD)
    # Then inv_fact[i] = inv_fact[i+1] * (i+1) mod MOD
    for i in range(X_max - 1, -1, -1): # loop down to i=0 to compute inv_fact[0]
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % self.MOD

    # nCr function using precomputed tables
    def nCr_mod_p(nn_val, rr_val):
        if rr_val < 0 or rr_val > nn_val: # Arguments out of standard range for combinations
            return 0
        # C(nn,rr) = nn! / (rr! * (nn-rr)!)
        numerator = fact[nn_val]
        denominator_part1 = inv_fact[rr_val]
        denominator_part2 = inv_fact[nn_val-rr_val]
        # Result is (numerator * den_part1 * den_part2) % MOD
        res = (numerator * denominator_part1) % self.MOD
        res = (res * denominator_part2) % self.MOD
        return res

    # Calculate C(m+1, 3)
    val_C_mplus1_3 = nCr_mod_p(m+1, 3)
    # Calculate S_r = n^2 * C(m+1, 3)
    term_S_r = (pow(n, 2, self.MOD) * val_C_mplus1_3) % self.MOD
    
    # Calculate C(n+1, 3)
    val_C_nplus1_3 = nCr_mod_p(n+1, 3)
    # Calculate S_c = m^2 * C(n+1, 3)
    term_S_c = (pow(m, 2, self.MOD) * val_C_nplus1_3) % self.MOD
    
    # S_dist_all_pairs_cells is the sum of Manhattan distances between all PAIRS OF DISTINCT CELLS in the grid
    S_dist_all_pairs_cells = (term_S_r + term_S_c) % self.MOD
    
    # For any specific pair of cells {A,B}, they are chosen to host pieces in C(N-2, k-2) arrangements.
    # The total sum is S_dist_all_pairs_cells * C(N-2, k-2)
    comb_factor_Nminus2_kminus2 = nCr_mod_p(N-2, k-2)
    
    total_sum = (S_dist_all_pairs_cells * comb_factor_Nminus2_kminus2) % self.MOD
    
    return total_sum