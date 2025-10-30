from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Generate all possible ways to divide the array into 3 subarrays
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate the cost of the current division
                cost = nums[0] + nums[i] + nums[j]
                min_cost = min(min_cost, cost)
        
        return min_cost