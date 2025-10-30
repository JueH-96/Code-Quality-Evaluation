from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        n = len(nums)
        max_distance = 0
        for i in range(n):
            # Compute absolute difference between current element and the next (circularly)
            diff = abs(nums[i] - nums[(i+1) % n])
            max_distance = max(max_distance, diff)
        return max_distance