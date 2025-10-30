from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # We need to split into three non-empty contiguous subarrays.
        # Let the first cut be after index i (0 <= i <= n-3),
        # and the second cut be after index j (i+1 <= j <= n-2).
        # Then the three segments are:
        #   nums[0..i], nums[i+1..j], nums[j+1..n-1]
        # The cost is nums[0] + nums[i+1] + nums[j+1].
        best = float('inf')
        for i in range(0, n-2):
            for j in range(i+1, n-1):
                cost = nums[0] + nums[i+1] + nums[j+1]
                if cost < best:
                    best = cost
        return best