import math
from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Compute prefix min array
        prefix_min = [0] * n
        prefix_min[0] = nums[0]
        for i in range(1, n):
            prefix_min[i] = min(prefix_min[i - 1], nums[i])
        
        # Compute suffix min array
        suffix_min = [0] * n
        suffix_min[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            suffix_min[i] = min(nums[i], suffix_min[i + 1])
        
        # Initialize min_sum to infinity
        min_sum = float('inf')
        
        # Iterate through each possible j index
        for j in range(1, n - 1):
            # Check if there is a valid i and k for this j
            if prefix_min[j - 1] < nums[j] and suffix_min[j + 1] < nums[j]:
                current_sum = prefix_min[j - 1] + nums[j] + suffix_min[j + 1]
                if current_sum < min_sum:
                    min_sum = current_sum
        
        # If no mountain triplet found, return -1
        if min_sum == float('inf'):
            return -1
        else:
            return min_sum