from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        """
        Calculates the minimum cost to collect chocolates of all types.
        """
        n = len(nums)
        
        # min_costs[i] stores the minimum cost to acquire chocolate of type i,
        # considering all shifts up to the current number of shifts.
        min_costs = list(nums)
        
        # ans is initialized with the cost for 0 shifts (no operation cost).
        ans = sum(min_costs)
        
        # Iterate through the number of shifts `k` from 1 to n-1.
        # It's never optimal to perform more than n-1 shifts.
        for k in range(1, n):
            # Update min_costs for each type based on the costs after k shifts.
            for i in range(n):
                # After k shifts, the chocolate of type `i` is the one that was
                # originally at index (i - k). The modulo operator handles wrapping around.
                # In Python, `(i - k) % n` correctly handles negative results.
                cost_after_k_shifts = nums[(i - k) % n]
                min_costs[i] = min(min_costs[i], cost_after_k_shifts)
            
            # Calculate the total cost if we perform k shifts.
            # This includes the operation cost and the sum of the best
            # collection costs found so far.
            op_cost = k * x
            collection_cost = sum(min_costs)
            total_cost = op_cost + collection_cost
            
            # Update the overall minimum cost.
            ans = min(ans, total_cost)
            
        return ans