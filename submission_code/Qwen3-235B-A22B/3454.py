from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute the difference array
        diff = [target[i] - nums[i] for i in range(n)]
        
        # Initialize operations count with the absolute value of the first element
        operations = abs(diff[0])
        
        # Sum the absolute differences between consecutive elements
        for i in range(1, n):
            operations += abs(diff[i] - diff[i-1])
        
        return operations