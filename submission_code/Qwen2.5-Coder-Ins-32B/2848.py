from typing import List
from functools import lru_cache

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        @lru_cache(None)
        def dp(mask, last):
            if mask == (1 << n) - 1:
                return 1
            count = 0
            for i in range(n):
                if not (mask & (1 << i)):
                    if nums[last] % nums[i] == 0 or nums[i] % nums[last] == 0:
                        count = (count + dp(mask | (1 << i), i)) % MOD
            return count
        
        total = 0
        for i in range(n):
            total = (total + dp(1 << i, i)) % MOD
        return total