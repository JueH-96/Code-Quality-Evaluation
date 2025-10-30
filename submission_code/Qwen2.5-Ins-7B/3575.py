from typing import List
from functools import lru_cache

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        ors = [0] * (n + 1)
        
        for i in range(1, n + 1):
            ors[i] = ors[i - 1] | nums[i - 1]
            for j in range(k, 0, -1):
                dp[i] = max(dp[i], dp[i - j] | (ors[i] ^ ors[i - j]))
        
        return dp[n]