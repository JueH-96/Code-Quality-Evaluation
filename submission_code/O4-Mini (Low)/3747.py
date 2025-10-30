from typing import List

class Solution:
    def maxAdjacentDistance(self, nums: List[int]) -> int:
        """
        Given a circular array nums, return the maximum absolute difference 
        between any two adjacent elements (considering the circular wrap).
        """
        n = len(nums)
        if n < 2:
            return 0  # trivial, though constraints guarantee n >= 2
        
        max_diff = 0
        # Check each pair (i, i+1) and also the wrap-around pair (n-1, 0)
        for i in range(n):
            j = (i + 1) % n
            diff = abs(nums[i] - nums[j])
            if diff > max_diff:
                max_diff = diff
        
        return max_diff