from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Precompute suffix minimums
        suffix_min = [0] * n
        suffix_min[n-1] = float('inf')  # No elements after the last index
        for i in range(n-2, -1, -1):
            suffix_min[i] = min(nums[i+1], suffix_min[i+1])
        
        # Initialize minimum sum to a large value
        min_sum = float('inf')
        
        # Iterate through possible split points
        for i in range(1, n-1):
            current_sum = nums[0] + nums[i] + suffix_min[i]
            if current_sum < min_sum:
                min_sum = current_sum
                
        return min_sum