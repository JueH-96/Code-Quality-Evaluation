class Solution:
    MOD = 10**9 + 7
    max_n = 10**5 + 10  # Sufficient for constraints
    fact = None
    inv_fact = None

    def comb(self, a, b):
        if b < 0 or b > a:
            return 0
        MOD = Solution.MOD
        return self.fact[a] * self.inv_fact[b] % MOD * self.inv_fact[a - b] % MOD

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        if m == 1:
            return 1 if (k == n - 1) else 0
        
        # Precompute fact and inv_fact if not already done
        if Solution.fact is None:
            max_n = Solution.max_n
            MOD = Solution.MOD
            fact = [1] * max_n
            for i in range(1, max_n):
                fact[i] = fact[i-1] * i % MOD
            inv_fact = [1] * max_n
            inv_fact[max_n - 1] = pow(fact[max_n - 1], MOD - 2, MOD)
            for i in range(max_n - 2, -1, -1):
                inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
            Solution.fact = fact
            Solution.inv_fact = inv_fact
        
        MOD = Solution.MOD
        r = n - k
        # Compute combination C(n-1, k)
        c = self.comb(n-1, k)
        # Compute term = m * (m-1)^(r-1) mod MOD
        exponent = r - 1
        power_term = pow(m-1, exponent, MOD)
        term = m * power_term % MOD
        return (c * term) % MOD