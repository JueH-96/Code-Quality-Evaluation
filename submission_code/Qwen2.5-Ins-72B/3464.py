from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # Calculate the cost of the entire array
        total_cost = nums[0]
        for i in range(1, n):
            total_cost += nums[i] * (-1) ** i
        
        # Calculate the maximum possible cost by considering splits
        max_cost = total_cost
        for i in range(1, n):
            # Calculate the cost if we split at index i
            split_cost = nums[i] * (-1) ** (i - 1) + nums[i] * (-1) ** i
            total_cost += split_cost
            max_cost = max(max_cost, total_cost)
        
        return max_cost