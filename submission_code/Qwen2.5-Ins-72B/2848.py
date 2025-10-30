from typing import List
from functools import lru_cache

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        @lru_cache(None)
        def dfs(index, prev, mask):
            if index == len(nums):
                return 1
            count = 0
            for i in range(len(nums)):
                if not mask & (1 << i) and (prev == -1 or nums[i] % prev == 0 or prev % nums[i] == 0):
                    count += dfs(index + 1, nums[i], mask | (1 << i))
            return count % MOD
        
        return dfs(0, -1, 0)