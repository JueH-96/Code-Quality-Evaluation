class Solution:
    mod = 10**9 + 7
    max_total = 2000
    fact = None
    inv_fact = None

    def valueAfterKSeconds(self, n: int, k: int) -> int:
        if Solution.fact is None:
            max_n = Solution.max_total
            mod = Solution.mod
            fact_arr = [1] * (max_n + 1)
            inv_fact_arr = [1] * (max_n + 1)
            for i in range(1, max_n + 1):
                fact_arr[i] = fact_arr[i - 1] * i % mod
            inv_fact_arr[max_n] = pow(fact_arr[max_n], mod - 2, mod)
            for i in range(max_n, 0, -1):
                inv_fact_arr[i - 1] = inv_fact_arr[i] * i % mod
            Solution.fact = fact_arr
            Solution.inv_fact = inv_fact_arr
        
        total = k + n - 1
        r = n - 1
        res = Solution.fact[total] * Solution.inv_fact[r] % Solution.mod
        res = res * Solution.inv_fact[total - r] % Solution.mod
        return res