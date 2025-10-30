from functools import lru_cache
from math import gcd
from itertools import permutations

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        @lru_cache(None)
        def dfs(mask, prev):
            if mask == (1 << len(nums)) - 1:
                return 1
            
            count = 0
            for i in range(len(nums)):
                if mask & (1 << i) == 0 and (nums[i] % prev == 0 or prev % nums[i] == 0):
                    count += dfs(mask | (1 << i), nums[i])
                    count %= MOD
            return count
        
        return sum(dfs(1 << i, nums[i]) for i in range(len(nums))) % MOD