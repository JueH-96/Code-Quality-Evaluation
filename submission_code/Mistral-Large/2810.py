from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        original_cost = sum(nums)

        # Initialize the minimum cost with the original cost
        min_cost = original_cost

        # Calculate the cost for each possible number of operations
        for operations in range(1, n):
            cost = operations * x
            for i in range(n):
                cost += nums[(i + operations) % n]
            min_cost = min(min_cost, cost)

        return min_cost