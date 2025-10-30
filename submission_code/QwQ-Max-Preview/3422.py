class Solution:
    MOD = 10**9 + 7

    def valueAfterKSeconds(self, n: int, k: int) -> int:
        max_precompute = 2000
        if not hasattr(Solution, 'fact'):
            # Precompute fact and inv_fact up to max_precompute
            fact = [1] * (max_precompute + 1)
            for i in range(1, max_precompute + 1):
                fact[i] = fact[i-1] * i % Solution.MOD
            inv_fact = [1] * (max_precompute + 1)
            inv_fact[max_precompute] = pow(fact[max_precompute], Solution.MOD - 2, Solution.MOD)
            for i in range(max_precompute - 1, -1, -1):
                inv_fact[i] = inv_fact[i + 1] * (i + 1) % Solution.MOD
            Solution.fact = fact
            Solution.inv_fact = inv_fact

        a = n + k - 1
        b = n - 1
        if b > a or b < 0:
            return 0
        return (Solution.fact[a] * Solution.inv_fact[b] % Solution.MOD) * Solution.inv_fact[a - b] % Solution.MOD