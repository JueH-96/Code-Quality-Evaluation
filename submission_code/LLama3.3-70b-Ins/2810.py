from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        total_cost = sum(nums)
        
        # Calculate the cost of performing the operation i times
        operation_cost = [0] * n
        for i in range(n):
            operation_cost[i] = x * i
        
        # Calculate the minimum cost to collect chocolates of all types
        min_cost = float('inf')
        for i in range(n):
            # Calculate the cost of collecting chocolates of all types after performing the operation i times
            cost = sum([nums[(j - i) % n] for j in range(n)])
            # Update the minimum cost
            min_cost = min(min_cost, cost + operation_cost[i])
        
        return min_cost