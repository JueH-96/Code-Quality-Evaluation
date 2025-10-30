from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array
        nums_sorted = sorted(nums)
        
        # Initialize minimum difference to a large value
        min_diff = float('inf')
        
        # Iterate through the sorted array and find the minimal adjacent difference
        for i in range(1, len(nums_sorted)):
            current_diff = nums_sorted[i] - nums_sorted[i-1]
            if current_diff < min_diff:
                min_diff = current_diff
                
        return min_diff