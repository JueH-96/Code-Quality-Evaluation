from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute factorial and inverse factorial modulo MOD up to n
        max_n = n
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        def comb(n_comb, k_comb):
            if k_comb < 0 or k_comb > n_comb:
                return 0
            return fact[n_comb] * inv_fact[k_comb] % MOD * inv_fact[n_comb - k_comb] % MOD
        
        res = 0
        for k in range(n + 1):
            cnt_less = bisect_left(nums, k)
            cnt_more = n - bisect_right(nums, k)
            if cnt_more == (n - k) and cnt_less >= k:
                res = (res + comb(cnt_less, k)) % MOD
        return res