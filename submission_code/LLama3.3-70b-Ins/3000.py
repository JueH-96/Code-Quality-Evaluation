from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        min_diff = float('inf')
        
        # Iterate over all pairs of elements in the array
        for i in range(len(nums)):
            for j in range(i + x, len(nums)):
                # Calculate the absolute difference between the current pair of elements
                diff = abs(nums[i] - nums[j])
                # Update the minimum difference if the current difference is smaller
                min_diff = min(min_diff, diff)
        
        return min_diff