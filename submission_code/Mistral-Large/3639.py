from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        # Calculate the sum of all elements in nums
        total_sum = sum(nums)

        # Calculate the total number of decrement operations
        total_decrements = 0
        for l, r in queries:
            total_decrements += (r - l + 1)

        # Check if the total sum of nums is equal to the total number of decrements
        return total_sum == total_decrements