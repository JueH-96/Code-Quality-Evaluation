from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Precompute the suffix minimum of nums
        suffix_min = [0] * n
        suffix_min[-1] = nums[-1]
        for i in range(n - 2, -1, -1):
            suffix_min[i] = min(nums[i], suffix_min[i + 1])
        
        # The cost of the first subarray is always nums[0].
        base = nums[0]
        ans = float('inf')
        
        # Try every split point i for the second subarray start (i from 1 to n-2)
        for i in range(1, n - 1):
            # The third subarray must start at j >= i+1, so its minimal first element
            # is suffix_min[i+1].
            cost = base + nums[i] + suffix_min[i + 1]
            ans = min(ans, cost)
        
        return ans