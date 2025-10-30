# Precomputation and helper function before the class definition
# This section should be placed before the Solution class definition.

MOD = 10**9 + 7
inv2 = pow(2, MOD - 2, MOD) # Modular inverse of 2
inv3 = pow(3, MOD - 2, MOD) # Modular inverse of 3

# Maximum possible value of m*n is 10^5. We need factorials up to m*n - 2.
N_max_fact = 100000
fact = [0] * (N_max_fact + 1)
invfact = [0] * (N_max_fact + 1)

# Precompute factorials modulo MOD
fact[0] = 1
for i in range(1, N_max_fact + 1):
    fact[i] = fact[i-1] * i % MOD

# Precompute inverse factorials modulo MOD using Fermat's Little Theorem
# invfact[i] = (i!)^(-1) mod MOD
invfact[N_max_fact] = pow(fact[N_max_fact], MOD - 2, MOD)
for i in range(N_max_fact - 1, -1, -1):
    # invfact[i] = invfact[i+1] * (i+1) mod MOD
    # Base case invfact[0] = 1!^(-1) = 1^(-1) = 1 mod MOD, already set.
    # The loop correctly computes invfact[0] from invfact[1] * 1
    invfact[i] = invfact[i+1] * (i+1) % MOD

def combinations(n, k):
    """Calculates nCk mod MOD using precomputed factorials and inverse factorials."""
    # Uses the globally defined fact, invfact, and MOD
    if k < 0 or k > n:
        return 0
    # Ensure indices are within bounds of precomputed arrays
    if n > N_max_fact:
         # This case should not be reached given the constraints m*n <= 10^5
         # and we are calculating C(N-2, k-2) where N=m*n.
         # If N > N_max_fact + 2, we would need to recompute or handle differently.
         # But N <= 10^5, so N-2 <= 99998 <= N_max_fact.
         return 0 # Should not happen

    return fact[n] * invfact[k] % MOD * invfact[n-k] % MOD

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        # Use the globally precomputed values and helper function
        # global MOD, inv2, inv3, fact, invfact, combinations # Not strictly needed for reading globals

        N = m * n

        # Calculate S_m = sum_{0 <= r1 < m, 0 <= r2 < m, r1 != r2} |r1 - r2| = (m-1)m(m+1)/3 mod MOD
        # (m-1) * m * (m+1) is divisible by 3. Modular inverse of 3 is correct.
        # m-1 >= 0 since m >= 1. m, m+1 >= 1. Simple % MOD is fine.
        m_minus_1_mod = (m - 1) % MOD
        m_mod = m % MOD
        m_plus_1_mod = (m + 1) % MOD
        m_term = m_minus_1_mod * m_mod % MOD
        m_term = m_term * m_plus_1_mod % MOD
        S_m = m_term * inv3 % MOD

        # Calculate S_n = sum_{0 <= c1 < n, 0 <= c2 < n, c1 != c2} |c1 - c2| = (n-1)n(n+1)/3 mod MOD
        # n-1 >= 0 since n >= 1. n, n+1 >= 1. Simple % MOD is fine.
        n_minus_1_mod = (n - 1) % MOD
        n_mod = n % MOD
        n_plus_1_mod = (n + 1) % MOD
        n_term = n_minus_1_mod * n_mod % MOD
        n_term = n_term * n_plus_1_mod % MOD
        S_n = n_term * inv3 % MOD

        # The total sum is the sum of distances between every unordered pair of distinct cells {cell1, cell2},
        # multiplied by the number of arrangements containing that pair of cells, which is C(N-2, k-2).
        # Sum over unordered pairs of distinct cells = 1/2 * Sum over ordered pairs of distinct cells
        # Sum_{(r1,c1)!=(r2,c2)} (|r1-r2| + |c1-c2|) = sum_{(r1,c1)!=(r2,c2)} |r1-r2| + sum_{(r1,c1)!=(r2,c2)} |c1-c2|
        # sum_{(r1,c1)!=(r2,c2)} |r1-r2| = n*n * S_m (sum over ordered distinct rows)
        # sum_{(r1,c1)!=(r2,c2)} |c1-c2| = m*m * S_n (sum over ordered distinct columns)
        # Sum over ordered pairs of distinct cells = (n*n * S_m + m*m * S_n) mod MOD

        n_sq = n * n % MOD
        m_sq = m * m % MOD

        sum_dist_all_pairs_ordered = (n_sq * S_m % MOD + m_sq * S_n % MOD) % MOD

        # Sum over all unordered pairs of distinct cells {cell1, cell2} is half
        sum_dist_all_pairs_unordered = sum_dist_all_pairs_ordered * inv2 % MOD

        # Each unordered pair of distinct cells {cell1, cell2} appears in C(N-2, k-2) arrangements.
        # Total Sum = sum_{{cell1, cell2}, cell1 != cell2} dist(cell1, cell2) * C(N-2, k-2)

        # Calculate C(N-2, k-2)
        # N = m*n. Constraints: 2 <= k <= N.
        # So k-2 >= 0 and k-2 <= N-2. Valid for combinations(N-2, k-2).
        comb_factor = combinations(N - 2, k - 2)

        total_sum = sum_dist_all_pairs_unordered * comb_factor % MOD

        return total_sum