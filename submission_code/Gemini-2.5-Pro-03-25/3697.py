import math
from typing import List

class Solution:
    """
    Solves the minimum increments problem using dynamic programming with bitmasking.
    The state dp[mask] represents the minimum total increments required to ensure that 
    for each target element corresponding to a set bit in 'mask', there exists at least 
    one element in the modified `nums` array that is a multiple of that target element.
    The transitions consider each element `n` from `nums` and calculate the cost 
    to modify `n` to cover various subsets of targets.
    """
    
    def gcd(self, a: int, b: int) -> int:
        """
        Computes the greatest common divisor (GCD) of two non-negative integers a and b.
        Uses the efficient `math.gcd` function.
        """
        # Constraints state target[i] >= 1, so inputs to gcd derived from targets will be positive.
        # If inputs could be zero or negative, math.gcd handles them appropriately.
        return math.gcd(a, b)

    def lcm(self, a: int, b: int) -> int:
        """
        Computes the least common multiple (LCM) of two positive integers a and b.
        """
        # Based on constraints target[i] >= 1, the inputs a and b will be positive integers
        # when calculating LCM of targets.
        if a == 0 or b == 0: 
             # This case should not be reached given constraints on target values.
             # LCM involving 0 is typically defined as 0.
             return 0
        # Calculates LCM using the formula: lcm(a, b) = (|a * b|) / gcd(a, b)
        # Python's integers handle arbitrary precision, preventing overflow for the product a * b.
        # Integer division // ensures the result is an integer.
        # abs() is not needed here as inputs are positive.
        return (a * b) // self.gcd(a, b)

    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        """
        Calculates the minimum total number of increments required on elements of `nums` 
        such that every element in `target` has at least one multiple in the modified `nums`.
        """
        k = len(target) # Number of target elements. Constraints: 1 <= k <= 4.
        
        # Initialize the DP table. dp[mask] stores the minimum cost (total increments)
        # to satisfy the condition for the subset of targets represented by the bitmask 'mask'.
        # The size of the DP table is 2^k.
        infinity = float('inf')
        dp = [infinity] * (1 << k)
        
        # Base case: The cost to cover an empty set of targets (mask 0) is 0, 
        # as no increments are needed.
        dp[0] = 0 

        # Memoization dictionary to store precomputed LCM values for subsets of targets.
        # This avoids redundant LCM calculations. Key: subset mask, Value: LCM of targets in the subset.
        memo_lcm = {} 

        # Precompute the LCM for all possible non-empty subsets of targets.
        # A subset is represented by a bitmask 'sub_mask' from 1 to 2^k - 1.
        for sub_mask in range(1, 1 << k):
            targets_in_subset = []
            # Identify the target elements corresponding to the set bits in 'sub_mask'.
            for j in range(k):
                if (sub_mask >> j) & 1: # Check if the j-th bit is set
                    targets_in_subset.append(target[j])
            
            # Calculate the LCM of the targets in the current subset iteratively.
            # Start with the first target in the subset (guaranteed to exist since sub_mask >= 1).
            current_lcm = targets_in_subset[0]
            # Sequentially compute LCM with the remaining targets in the subset.
            for i in range(1, len(targets_in_subset)):
                current_lcm = self.lcm(current_lcm, targets_in_subset[i])
                # LCM can grow large, but Python's integers handle it. Max value up to ~10^16.
            
            # Store the computed LCM for this subset mask in the memoization table.
            memo_lcm[sub_mask] = current_lcm

        # Iterate through each number 'n' in the input array 'nums'.
        # For each 'n', we consider how it can contribute to satisfying the target requirements.
        for n in nums:
            # Calculate the cost (number of increments) required to make the current number 'n' 
            # become a multiple of the LCM for each non-empty subset of targets.
            costs_for_n = {} # Store costs associated with 'n' covering various subsets.
            for sub_mask in range(1, 1 << k):
                # Retrieve the precomputed LCM for this target subset 'sub_mask'.
                L = memo_lcm[sub_mask]
                
                # Calculate the increments needed: `cost = (L - (n % L)) % L`.
                # This formula finds the smallest non-negative increment `inc` such that `(n + inc)` is divisible by `L`.
                # If `n` is already a multiple of `L`, `n % L` is 0, and `cost` becomes `(L - 0) % L = 0`.
                rem = n % L
                cost_n_s = (L - rem) % L 
                
                # Store the calculated cost for using 'n' to cover the subset 'sub_mask'.
                costs_for_n[sub_mask] = cost_n_s

            # Update the DP table based on the possibilities offered by the current number 'n'.
            # We use a temporary copy `next_dp` to store the results of this iteration's updates,
            # preventing updates from affecting calculations within the same iteration (for the same 'n').
            next_dp = list(dp) 
            
            # Iterate through all previously reached states (masks) represented in the current `dp` table.
            for current_mask in range(1 << k):
                # If the state `current_mask` is currently unreachable (cost is infinity), skip it.
                if dp[current_mask] == infinity:
                    continue
                
                # Consider using the current number `n` to cover an additional subset `sub_mask` of targets.
                # We iterate through all non-empty subsets `sub_mask`.
                for sub_mask in range(1, 1 << k):
                    # Get the precalculated cost for `n` to cover `sub_mask`.
                    cost_n_s = costs_for_n[sub_mask]
                    
                    # Determine the resulting state (mask) if we use `n` to cover `sub_mask`
                    # starting from the state `current_mask`. The new mask combines targets covered by
                    # `current_mask` and `sub_mask` using bitwise OR.
                    new_mask = current_mask | sub_mask
                    
                    # Calculate the total cost to reach this `new_mask` via the current path:
                    # cost of reaching `current_mask` + cost of using `n` to cover `sub_mask`.
                    potential_new_cost = dp[current_mask] + cost_n_s
                    
                    # Update the minimum cost recorded for `new_mask` in `next_dp` if this path is cheaper
                    # than any previously found path to `new_mask`.
                    if potential_new_cost < next_dp[new_mask]:
                         next_dp[new_mask] = potential_new_cost

            # After considering all possible uses of 'n' starting from all current states,
            # update the main DP table `dp` with the results stored in `next_dp` for the next iteration (next number in `nums`).
            dp = next_dp

        # The final answer is the minimum cost stored in the DP state corresponding to the mask
        # where all target elements are covered. This mask has all k bits set, represented by `(1 << k) - 1`.
        final_cost = dp[(1 << k) - 1]
        
        # The problem constraints and the nature of the operation (incrementing) imply that
        # a solution is always possible and the cost will be finite.
        # Return the final minimum cost as an integer.
        # If `final_cost` remained `infinity`, it would indicate impossibility (or an issue), 
        # but this is not expected here.
        return int(final_cost)