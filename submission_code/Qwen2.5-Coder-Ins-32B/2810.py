from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        for k in range(n):
            current_cost = k * x
            for i in range(n):
                current_cost += min(nums[(i - j) % n] for j in range(k + 1))
            min_cost = min(min_cost, current_cost)
        
        return min_cost