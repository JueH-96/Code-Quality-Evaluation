from typing import List
from functools import lru_cache

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        @lru_cache(None)
        def dp(i, k, x, or1):
            if k == 0:
                return x ^ or1
            if i == len(nums):
                return -float('inf')
            # Skip current number
            res = dp(i + 1, k, x, or1)
            # Include current number in the first half
            if k > x:
                res = max(res, dp(i + 1, k, x + 1, or1 | nums[i]))
            # Include current number in the second half
            else:
                res = max(res, dp(i + 1, k, x, or1 | nums[i]))
            return res
        
        return dp(0, k, 0, 0)