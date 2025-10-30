class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        N = m * n

        # According to constraints, 2 <= k <= m*n.
        # This implies N >= 2 and N-2 >= k-2 >= 0.
        # So C(N-2, k-2) is well-defined.

        # We need to compute C(N-2, k-2) mod MOD.
        # This requires precomputing factorials up to N.
        max_fact = N
        fact = [1] * (max_fact + 1)
        invFact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Modular inverse using Fermat's Little Theorem
        invFact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            invFact[i] = (invFact[i+1] * (i + 1)) % MOD

        def nCr_mod(n_val, r_val):
            if r_val < 0 or r_val > n_val:
                return 0
            # n! / (k! * (n-k)!)
            numerator = fact[n_val]
            denominator = (invFact[r_val] * invFact[n_val - r_val]) % MOD
            return (numerator * denominator) % MOD

        # The total sum is C(N - 2, k - 2) * (Sum of distances between all cell pairs).
        
        # Part 1: C(N-2, k-2)
        # This is the number of ways to place the remaining k-2 pieces
        # after choosing two specific cells to be occupied.
        num_arrangements_for_pair = nCr_mod(N - 2, k - 2)

        # Part 2: Sum of Manhattan distances between all pairs of cells.
        # The sum of distances between all cell pairs can be shown to be:
        # (m*n/6) * [n(m^2-1) + m(n^2-1)] = (m*n * (m+n) * (m*n - 1)) / 6
        
        # We calculate this term modularly.
        term_mn = (m * n) % MOD
        term_sum_mn = (m + n) % MOD
        term_mn_minus_1 = (m * n - 1) % MOD
        inv6 = pow(6, MOD - 2, MOD)
        
        # Total distance sum for all cell pairs
        total_cell_pair_dist = (term_mn * term_sum_mn) % MOD
        total_cell_pair_dist = (total_cell_pair_dist * term_mn_minus_1) % MOD
        total_cell_pair_dist = (total_cell_pair_dist * inv6) % MOD

        # Final result is the product of the two parts.
        result = (num_arrangements_for_pair * total_cell_pair_dist) % MOD
        
        return result