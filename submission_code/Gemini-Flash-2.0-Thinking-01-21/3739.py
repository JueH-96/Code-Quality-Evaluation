import sys

# sys.setrecursionlimit(200000) # Not strictly needed for iterative power and factorial precomputation

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Maximum value for m*n is 10^5.
        # We need combinations C(N-2, k-2). The largest index in factorial will be max(N-2, k-2, N-k).
        # N = m*n <= 10^5. So N-2 <= 10^5 - 2.
        # Given constraints 2 <= k <= m*n, we have k-2 >= 0 and N-k >= 0.
        # max(k-2, N-k) <= N-2.
        # So max index needed for factorials is N-2.
        # Precompute up to 10^5 is sufficient as N <= 10^5.
        MAX_VAL = 10**5

        fact = [1] * (MAX_VAL + 1)
        invFact = [1] * (MAX_VAL + 1)

        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res

        def inverse(a):
            # Modular inverse exists only if gcd(a, MOD) = 1.
            # MOD is prime, so inverse exists for all a != 0 (mod MOD).
            # This case should not happen for numbers we need inverse of (factorials, 6)
            return power(a, MOD - 2)

        def precompute_factorials(max_val):
            # fact[0] = 1, invFact[0] = 1 (by convention 0! = 1)
            for i in range(1, max_val + 1):
                fact[i] = (fact[i - 1] * i) % MOD
            # Need inverse up to max_val
            invFact[max_val] = inverse(fact[max_val])
            for i in range(max_val - 1, 0, -1):
                invFact[i] = (invFact[i + 1] * (i + 1)) % MOD
            # invFact[0] = 1

        def combinations(n_val, k_val):
            if k_val < 0 or k_val > n_val:
                return 0
            # Requires fact and invFact up to n_val.
            # n_val = N-2 <= 10^5-2, which is covered by MAX_VAL = 10^5.
            return (fact[n_val] * invFact[k_val] % MOD * invFact[n_val - k_val] % MOD)

        precompute_factorials(MAX_VAL)

        N = m * n
        
        # The problem states 2 <= k <= m*n and 2 <= m*n.
        # So N >= 2 and k >= 2 and k <= N.
        # Thus N-2 >= 0 and k-2 >= 0 and N-k >= 0.
        # C(N-2, k-2) is well-defined.

        # Sum of Manhattan distances between all unordered pairs of distinct cells {L_1, L_2}.
        # S_cells = \sum_{0 <= i < j < N} dist(cell_i, cell_j)
        # This sum is n^2 * m(m^2-1)/6 + m^2 * n(n^2-1)/6 mod M
        # = n^2 * m(m-1)(m+1)/6 + m^2 * n(n-1)(n+1)/6 mod M
        
        m_mod = m % MOD
        n_mod = n % MOD
        
        inv6 = inverse(6)

        # Calculate m(m-1)(m+1)/6 mod M
        # m * (m-1) * (m+1) mod M
        # Use long long or ensure modulo at each step to prevent overflow
        m_term_num = (m_mod * ((m_mod - 1 + MOD) % MOD)) % MOD
        m_term_num = (m_term_num * ((m_mod + 1) % MOD)) % MOD
        m_term_div_6 = (m_term_num * inv6) % MOD
        
        # Calculate n(n-1)(n+1)/6 mod M
        # n * (n-1) * (n+1) mod M
        n_term_num = (n_mod * ((n_mod - 1 + MOD) % MOD)) % MOD
        n_term_num = (n_term_num * ((n_mod + 1) % MOD)) % MOD
        n_term_div_6 = (n_term_num * inv6) % MOD
        
        # Calculate n^2 * m(m^2-1)/6 mod M
        term1 = (n_mod * n_mod % MOD) * m_term_div_6 % MOD
        
        # Calculate m^2 * n(n^2-1)/6 mod M
        term2 = (m_mod * m_mod % MOD) * n_term_div_6 % MOD
        
        S_cells = (term1 + term2) % MOD
        
        # The number of arrangements containing any fixed pair of cells {L_1, L_2} is C(N-2, k-2)
        # Summing over all pairs of cells {L_1, L_2}, each contributes dist(L_1, L_2)
        # to C(N-2, k-2) arrangements.
        comb = combinations(N - 2, k - 2)
        
        # Total sum = C(N-2, k-2) * S_cells mod M
        result = (comb * S_cells) % MOD
        
        return result