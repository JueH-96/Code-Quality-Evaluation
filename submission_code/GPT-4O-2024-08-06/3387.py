from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        
        # Find the median index
        median_index = (n - 1) // 2
        
        # Calculate the number of operations needed to make the median equal to k
        operations = abs(nums[median_index] - k)
        
        return operations