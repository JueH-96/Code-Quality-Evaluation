import math
from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # min_cost_for_type[j] will store the minimum cost found so far for collecting chocolate of type j.
        # Initially, this is just the direct cost nums[j] (0 rotations).
        min_cost_for_type = list(nums)
        
        # overall_min_cost stores the minimum total cost across all possible number of rotations (k).
        # We initialize it with a very large value.
        overall_min_cost = float('inf')
        
        # We iterate k from 0 to n-1. k represents the total number of operations performed.
        # After n operations, the state of chocolate types returns to the initial state,
        # so checking k >= n operations would be redundant (e.g., n operations is equivalent to 0 operations
        # but with an added cost of n*x).
        for k in range(n):
            # Calculate the cost incurred by performing k operations.
            cost_of_operations = k * x
            
            # For each type j, its current effective cost (if acquired after k rotations)
            # is nums[(j - k + n) % n].
            # We update min_cost_for_type[j] to ensure it stores the minimum cost for type j
            # considering all possibilities from 0 to k rotations.
            # This update happens inside the loop for k, so by the time we sum min_cost_for_type,
            # it already reflects the best prices up to 'k' rotations.
            for j in range(n):
                # The chocolate currently of type 'j' was originally of type '(j - k + n) % n'.
                # Its cost is nums[(j - k + n) % n].
                # We update the minimum cost for type 'j' to be the minimum of its current
                # best price and this newly discovered price.
                min_cost_for_type[j] = min(min_cost_for_type[j], nums[(j - k + n) % n])
            
            # Calculate the sum of minimum costs for all chocolate types found up to k rotations.
            total_chocolate_cost = sum(min_cost_for_type)
            
            # The total cost for this specific number of operations (k) is the sum of
            # operation costs and the sum of minimum chocolate costs.
            current_total_cost = cost_of_operations + total_chocolate_cost
            
            # Update the overall minimum cost if this current total cost is lower.
            overall_min_cost = min(overall_min_cost, current_total_cost)
            
        return overall_min_cost