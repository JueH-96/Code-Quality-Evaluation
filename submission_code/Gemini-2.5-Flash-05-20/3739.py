import math

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Total number of cells in the grid
        N_cells = m * n
        
        # Precompute factorials and inverse factorials up to N_cells
        # for efficient calculation of combinations (nCr_mod_p).
        fact = [1] * (N_cells + 1)
        invFact = [1] * (N_cells + 1)
        
        # Calculate factorials
        for i in range(2, N_cells + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Calculate modular inverse of fact[N_cells] using Fermat's Little Theorem
        # a^(p-2) % p is the modular inverse of a for prime p
        invFact[N_cells] = pow(fact[N_cells], MOD - 2, MOD)
        
        # Calculate inverse factorials for i from N_cells-1 down to 0
        # invFact[i-1] = invFact[i] * i % MOD
        for i in range(N_cells - 1, 1, -1):
            invFact[i] = (invFact[i+1] * (i+1)) % MOD
            
        # Helper function to calculate nCr % MOD
        def nCr_mod_p(n, r):
            if r < 0 or r > n:
                return 0
            if r == 0 or r == n:
                return 1
            # nCr = n! / (r! * (n-r)!)
            numerator = fact[n]
            denominator = (invFact[r] * invFact[n-r]) % MOD
            return (numerator * denominator) % MOD

        # Step 1: Calculate the number of ways to choose the remaining k-2 pieces
        # from N_cells-2 cells, given that two specific cells are occupied.
        # This is C(N_cells - 2, k - 2).
        # Constraints: k >= 2 and k <= N_cells.
        # This implies k-2 >= 0 and N_cells-2 >= k-2,
        # so arguments to nCr_mod_p are always valid.
        combinations = nCr_mod_p(N_cells - 2, k - 2)

        # Step 2: Calculate the sum of Manhattan distances between every unordered pair
        # of distinct cells on the entire m x n grid.
        # The formula derived is (m*n/6) * (m+n) * (m*n-1).
        
        # Modular inverse of 6
        inv6 = pow(6, MOD - 2, MOD)
        
        term_mn = (m * n) % MOD
        term_m_plus_n = (m + n) % MOD
        # Ensure result is non-negative before modulo if term_mn - 1 is -1 (e.g., m*n=0 or 1)
        # However, N_cells = m*n >= 2 by constraints, so term_mn >= 2.
        term_mn_minus_1 = (term_mn - 1 + MOD) % MOD 
        
        # Calculate sum_dist_pairs_on_grid = (term_mn * term_m_plus_n * term_mn_minus_1) * inv6 % MOD
        sum_dist_pairs_on_grid = (term_mn * term_m_plus_n) % MOD
        sum_dist_pairs_on_grid = (sum_dist_pairs_on_grid * term_mn_minus_1) % MOD
        sum_dist_pairs_on_grid = (sum_dist_pairs_on_grid * inv6) % MOD

        # Step 3: Multiply the combinations with the sum of distances
        # and take modulo MOD.
        total_sum_manhattan = (combinations * sum_dist_pairs_on_grid) % MOD
        
        return total_sum_manhattan