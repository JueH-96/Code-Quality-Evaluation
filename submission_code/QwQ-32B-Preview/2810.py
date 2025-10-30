from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        current_min = nums.copy()
        total = sum(current_min)
        min_cost = total
        
        for k in range(1, n):
            for j in range(n):
                # Update the minimum cost for type j after k shifts
                shifted_index = (j - k) % n
                current_min[j] = min(current_min[j], nums[shifted_index])
            # Calculate the sum of current minimum costs
            total = sum(current_min)
            # Calculate the total cost including shift operations
            total_cost = total + k * x
            # Update the minimum cost if the current total is lower
            if total_cost < min_cost:
                min_cost = total_cost
                
        return min_cost