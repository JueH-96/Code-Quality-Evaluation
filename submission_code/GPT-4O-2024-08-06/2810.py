from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Initialize the minimum cost to a large number
        min_cost = float('inf')
        
        # Try to collect chocolates starting from each type
        for start in range(n):
            current_cost = 0
            # Calculate the cost of collecting chocolates starting from 'start' type
            for i in range(n):
                # Calculate the index of the chocolate type after i operations
                index = (start + i) % n
                # Add the minimum cost of the chocolate at this index
                current_cost += nums[index]
                # Add the cost of performing the operation x times
                if i < n - 1:
                    current_cost += x
            
            # Update the minimum cost found
            min_cost = min(min_cost, current_cost)
        
        return min_cost