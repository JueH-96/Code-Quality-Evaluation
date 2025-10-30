from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = [float('inf')] * n
        result = float('inf')
        
        for rotations in range(n):
            current_cost = rotations * x
            for i in range(n):
                min_cost[i] = min(min_cost[i], nums[(i + rotations) % n])
                current_cost += min_cost[i]
            result = min(result, current_cost)
        
        return result