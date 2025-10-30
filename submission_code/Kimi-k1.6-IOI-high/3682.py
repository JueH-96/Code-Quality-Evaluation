class Solution:
    MOD = 10**9 + 7
    max_precomputed = 0
    fact = []
    inv_fact = []
    precomputed = False

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        if not Solution.precomputed:
            max_n = 10**5
            Solution.fact = [1] * (max_n + 1)
            for i in range(1, max_n + 1):
                Solution.fact[i] = Solution.fact[i-1] * i % Solution.MOD
            Solution.inv_fact = [1] * (max_n + 1)
            Solution.inv_fact[max_n] = pow(Solution.fact[max_n], Solution.MOD-2, Solution.MOD)
            for i in range(max_n - 1, -1, -1):
                Solution.inv_fact[i] = Solution.inv_fact[i+1] * (i+1) % Solution.MOD
            Solution.precomputed = True
            Solution.max_precomputed = max_n

        # Edge case when k exceeds possible transitions or n is invalid
        if k < 0 or k > n - 1:
            return 0
        if n - 1 > Solution.max_precomputed:
            return 0

        # Compute combination C(n-1, k)
        comb = Solution.fact[n-1] * Solution.inv_fact[k] % Solution.MOD
        comb = comb * Solution.inv_fact[n-1 - k] % Solution.MOD

        # Compute the result
        res = comb * m % Solution.MOD
        exponent = n - k - 1
        if m == 1:
            pow_part = 1 if exponent == 0 else 0
        else:
            pow_part = pow(m-1, exponent, Solution.MOD)
        res = res * pow_part % Solution.MOD
        return res