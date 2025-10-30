from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        best = float('inf')
        # Iterate over every valid splitting index for the second subarray
        for i in range(1, n - 1):
            # Iterate over every valid splitting index for the third subarray (which comes after i)
            for j in range(i + 1, n):
                cost = nums[0] + nums[i] + nums[j]
                best = min(best, cost)
        return best