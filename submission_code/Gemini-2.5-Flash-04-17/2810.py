from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # min_costs[i] stores the minimum cost found so far to acquire type i
        # After 0 operations, type i is at index i, cost is nums[i].
        # Use Python's int type which handles large numbers.
        min_costs = list(nums)

        # min_overall_cost stores the minimum total cost found over different strategies (total number of operations performed).
        # The initial strategy is 0 operations: cost is sum of initial prices.
        min_overall_cost = sum(min_costs)

        # Iterate through the total number of operations k from 1 to n-1.
        # We consider strategies where we perform exactly k operations in total.
        # For a strategy involving k total operations, the cost of operations is k * x.
        # The cost of chocolates is the sum of the minimum costs to acquire each type,
        # considering the configurations after 0, 1, ..., up to k operations.
        for k in range(1, n):
            # Create a temporary list to calculate minimum costs considering up to k operations.
            # next_min_costs[i] will store the minimum cost for type i considering states 0 to k.
            # Initialize with minimum costs considering states 0 to k-1 (which is the current min_costs).
            next_min_costs = list(min_costs)
            
            # After k operations, the chocolate initially at index j (with cost nums[j]) is now of type (j + k) % n.
            # This means nums[j] is a potential cost for type (j + k) % n in the state after k operations.
            # We update the minimum cost for type (j + k) % n considering states up to k.
            # The minimum cost for type (j + k) % n considering states 0 to k is
            # min(minimum cost considering states 0 to k-1, cost in state k)
            # The minimum cost considering states 0 to k-1 is min_costs[(j + k) % n].
            # The cost in state k for type (j + k) % n is nums[j].
            
            # Iterate through each initial index j (0 to n-1)
            for j in range(n):
                 current_type_after_k_ops = (j + k) % n
                 cost_at_initial_index_j = nums[j]
                 
                 # Update the minimum cost for the type that this chocolate becomes after k operations.
                 # Compare the existing minimum cost for this type (from states 0..k-1)
                 # with the cost of this specific chocolate (which becomes this type in state k).
                 next_min_costs[current_type_after_k_ops] = min(min_costs[current_type_after_k_ops], cost_at_initial_index_j)
                 
            # Update min_costs for the next iteration (min_costs now holds minimums after considering up to k operations).
            min_costs = next_min_costs
            
            # Calculate the total cost for the strategy of performing k operations in total.
            # This cost is the sum of the minimum costs to acquire each type (considering up to k ops)
            # plus the cost of the k operations.
            current_total_cost = sum(min_costs) + k * x
            
            # Update the overall minimum total cost found so far.
            min_overall_cost = min(min_overall_cost, current_total_cost)

        return min_overall_cost