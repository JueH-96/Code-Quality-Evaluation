from typing import List
from sortedcontainers import SortedList

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [0] * n
        indices = SortedList(range(n))
        
        for i in range(n - 1, -1, -1):
            valid_indices = indices.bisect_left(i)
            while valid_indices < len(indices):
                j = indices[valid_indices]
                if target >= nums[j] - nums[i] >= -target:
                    dp[i] = max(dp[i], dp[j] + 1)
                else:
                    break
            indices.discard(i)
        
        return dp[0] if dp[0] > 0 else -1