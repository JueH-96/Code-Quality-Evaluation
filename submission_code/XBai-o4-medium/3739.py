class Solution:
    MOD = 10**9 + 7
    
    def distanceSum(self, m: int, n: int, k: int) -> int:
        # Precompute fact and inv_fact if not already done
        if not hasattr(Solution, 'fact'):
            max_n = 10**5
            Solution.fact = [1] * (max_n + 1)
            for i in range(1, max_n + 1):
                Solution.fact[i] = Solution.fact[i-1] * i % Solution.MOD
            Solution.inv_fact = [1] * (max_n + 1)
            Solution.inv_fact[max_n] = pow(Solution.fact[max_n], Solution.MOD-2, Solution.MOD)
            for i in range(max_n - 1, -1, -1):
                Solution.inv_fact[i] = Solution.inv_fact[i+1] * (i + 1) % Solution.MOD
        
        MOD = Solution.MOD
        
        # Compute combination C(a, b)
        t = m * n
        a = t - 2
        b = k - 2
        if b < 0 or b > a:
            return 0
        comb = Solution.fact[a] * Solution.inv_fact[b] % MOD
        comb = comb * Solution.inv_fact[a - b] % MOD
        
        # Compute term1: (m^3 - m) * n^2
        m3 = m * m % MOD
        m3 = m3 * m % MOD
        m3_minus_m = (m3 - m) % MOD
        n_sq = n * n % MOD
        term1 = m3_minus_m * n_sq % MOD
        
        # Compute term2: (n^3 - n) * m^2
        n3 = n * n % MOD
        n3 = n3 * n % MOD
        n3_minus_n = (n3 - n) % MOD
        m_sq = m * m % MOD
        term2 = n3_minus_n * m_sq % MOD
        
        sum_terms = (term1 + term2) % MOD
        inv6 = pow(6, MOD - 2, MOD)
        manhattan_total = sum_terms * inv6 % MOD
        
        total = manhattan_total * comb % MOD
        return total