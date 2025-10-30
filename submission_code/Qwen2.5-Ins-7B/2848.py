from functools import lru_cache

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        @lru_cache(None)
        def dfs(index, mask):
            if index == len(nums):
                return 1
            count = 0
            for i in range(len(nums)):
                if mask & (1 << i) == 0 and (nums[i] % nums[index] == 0 or nums[index] % nums[i] == 0):
                    count += dfs(index + 1, mask | (1 << i))
                    count %= MOD
            return count
        
        return dfs(0, 0)