from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array so that the minimal difference between any two entries
        # occurs between consecutive elements.
        nums.sort()
        
        # Initialize the minimum difference to a very large number.
        min_diff = float('inf')
        
        # Iterate through consecutive pairs in the sorted list.
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i - 1]
            if diff < min_diff:
                min_diff = diff
        
        return min_diff