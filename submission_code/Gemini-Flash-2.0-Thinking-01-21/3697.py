from typing import List
import math

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        k = len(target)
        
        # Helper to compute GCD using Python's built-in
        def gcd(a, b):
            return math.gcd(a, b)

        # Helper to compute LCM of a list of numbers
        # Uses arbitrary precision integers in Python.
        def list_lcm(lst):
            # Constraints: target[i] >= 1. So list is non-empty and contains positive integers.
            # LCM will be >= 1.
            res_lcm = 1
            for x in lst:
                # lcm(a, b) = (a * b) // gcd(a, b)
                # Using the form (a // gcd(a, b)) * b to compute lcm(a, b)
                # This avoids potential intermediate product overflow compared to (a * b) // gcd(a, b)
                # if not using arbitrary precision integers (although Python has them).
                common = gcd(res_lcm, x)
                res_lcm = (res_lcm // common) * x
            return res_lcm
            
        # Precompute LCM for all non-empty subsets of target
        # lcm_vals[mask] stores the LCM of target[j] for all j where the j-th bit of mask is set.
        # Size 2^k. Max k=4, so max size 16.
        lcm_vals = [0] * (1 << k)
        for mask in range(1, 1 << k):
            subset_target = [target[j] for j in range(k) if (mask >> j) & 1]
            lcm_val = list_lcm(subset_target)
            # If LCM is 0 (should not happen with positive inputs) or 1, handle accordingly.
            # A target value of 1 means any number is a multiple, cost is 0.
            # LCM of positive numbers is >= 1.
            lcm_vals[mask] = lcm_val

        # Precompute minimum cost to make any element in nums a multiple of lcm(subset) represented by mask
        # min_cost_for_subset[mask] = min_{n in nums} cost_to_make_n_multiple_of_lcm_vals[mask]
        # Size 2^k. Max size 16.
        min_cost_for_subset = [float('inf')] * (1 << k)
        for mask in range(1, 1 << k):
            L = lcm_vals[mask]
            # L will be >= 1 since target[i] >= 1 and mask is non-empty.
            
            current_min_cost = float('inf')
            # Iterate through all numbers in nums to find the minimum cost
            for n in nums:
                 # Cost to make n a multiple of L is (L - (n % L)) % L
                 # This is 0 if n is already a multiple, otherwise it's the difference to the next multiple.
                 cost = (L - (n % L)) % L
                 current_min_cost = min(current_min_cost, cost)
                 
            min_cost_for_subset[mask] = current_min_cost

        # DP table: dp[mask] is the minimum total operations required so that for every target[j]
        # where the j-th bit is set in mask, there exists at least one multiple of target[j]
        # in the modified nums array.
        # Size 2^k. Max size 16.
        dp = [float('inf')] * (1 << k)
        dp[0] = 0 # Base case: 0 operations to satisfy no requirements

        # Iterate through masks from 0 up to (1 << k) - 1
        # We only need to compute for masks > 0, as dp[0] is base case.
        # The order ensures dp[prev_mask] is computed before dp[mask] if prev_mask < mask.
        # This order (0 to 2^k-1) guarantees that.
        for mask in range(1, 1 << k):
            # To compute dp[mask], we consider the last "unit" of requirements satisfied.
            # This unit is a non-empty subset of 'mask', represented by 'submask'.
            # We assume that requirements in 'mask ^ submask' were already satisfied optimally
            # (minimum cost dp[mask ^ submask]), and we now incur the minimum cost to satisfy
            # the requirements in 'submask' using one *additional* element from nums.
            # The minimum cost to satisfy requirements in 'submask' using one element from nums
            # is precomputed as min_cost_for_subset[submask].
            
            # Iterate through all non-empty submasks of the current mask.
            # submask = mask will be the first iteration, covering dp[0] + min_cost_for_subset[mask]
            submask = mask
            while submask > 0:
                 prev_mask = mask ^ submask # The requirements satisfied before considering 'submask'
                
                 # Check if the previous state is reachable and the cost for the new requirements is finite
                 if dp[prev_mask] != float('inf') and min_cost_for_subset[submask] != float('inf'):
                    # The cost to reach 'mask' via this path is the cost to reach 'prev_mask'
                    # plus the minimum cost to satisfy 'submask' using one element from nums.
                    current_cost = dp[prev_mask] + min_cost_for_subset[submask]
                    
                    # Update the minimum cost for the current mask
                    dp[mask] = min(dp[mask], current_cost)
                    
                 # Move to the next smaller submask
                 submask = (submask - 1) & mask

        # The final answer is the minimum cost to satisfy all requirements (mask with all bits set to 1)
        # This is dp[(1<<k)-1] which represents the mask with all bits set to 1.
        return dp[(1 << k) - 1]