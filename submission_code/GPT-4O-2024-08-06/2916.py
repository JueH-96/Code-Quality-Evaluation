from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        
        # If the array has only one or two elements, we can always split it into n arrays.
        if n <= 2:
            return True
        
        # Check if there is any adjacent pair whose sum is >= m
        for i in range(n - 1):
            if nums[i] + nums[i + 1] >= m:
                return True
        
        # If no such pair exists, we cannot split the array into n arrays
        return False