from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        if not nums:
            return 0
        even = -float('inf')
        odd = nums[0]
        for i in range(1, len(nums)):
            num = nums[i]
            new_even = odd - num
            new_odd = max(even, odd) + num
            even, odd = new_even, new_odd
        return max(even, odd)