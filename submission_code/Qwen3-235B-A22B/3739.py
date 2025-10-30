class Solution:
    MOD = 10**9 + 7
    max_fact = 10**5
    fact = None
    inv_fact = None

    def precompute(self):
        if Solution.fact is not None:
            return
        max_n = Solution.max_fact
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % Solution.MOD
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], Solution.MOD - 2, Solution.MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % Solution.MOD
        Solution.fact = fact
        Solution.inv_fact = inv_fact

    def distanceSum(self, m: int, n: int, k: int) -> int:
        self.precompute()
        mod = Solution.MOD
        inv_6 = pow(6, mod - 2, mod)

        mn = m * n
        if k < 2 or k > mn:
            return 0

        # Compute A = n^2 * m * (m^2 - 1)
        n_sq = (n * n) % mod
        m_mod = m % mod
        m_sq_minus_1 = (m * m - 1) % mod
        A = (n_sq * m_mod) % mod
        A = (A * m_sq_minus_1) % mod

        # Compute B = m^2 * n * (n^2 - 1)
        m_sq = (m * m) % mod
        n_mod = n % mod
        n_sq_minus_1 = (n * n - 1) % mod
        B = (m_sq * n_mod) % mod
        B = (B * n_sq_minus_1) % mod

        sum_ab = (A + B) % mod
        s_grid = (sum_ab * inv_6) % mod

        # Compute combination C(mn-2, k-2)
        a = mn - 2
        b = k - 2
        if b < 0 or b > a:
            return 0
        comb = Solution.fact[a] * Solution.inv_fact[b] % mod
        comb = comb * Solution.inv_fact[a - b] % mod

        ans = (comb * s_grid) % mod
        return ans