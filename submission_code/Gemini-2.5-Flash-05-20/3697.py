import math
from typing import List

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        M = len(target)
        
        # dp[mask] stores the minimum total operations to satisfy the set of
        # target elements represented by 'mask'.
        # Initialize dp with infinity, dp[0] = 0 (no operations for no requirements).
        dp = [float('inf')] * (1 << M)
        dp[0] = 0
        
        # Upper bound for total possible operations, used for pruning unfeasible LCMs.
        # This is based on N_max * T_max = 5*10^4 * 10^4 = 5*10^8.
        # Adding max(nums) to this bound means that even if a single n_prime value is very high,
        # it might still be achievable if its cost is within this bound.
        # e.g., if n=1, and total ops is 5*10^8, n_prime could be 5*10^8.
        # So any LCM (which would become the needed_val for n_prime) significantly above this value
        # would result in a cost higher than the maximum possible total operations, making it non-optimal.
        MAX_FEASIBLE_NUM_VALUE = (5 * 10**8) + max(nums) 
        
        for n_val in nums:
            # Create a new dp array for this iteration to correctly apply updates.
            # This ensures that each 'n_val' is considered as a new resource,
            # building upon the 'dp' states computed from previous 'nums' elements.
            new_dp = list(dp) 
            
            # Iterate 'mask' in reverse (from (1<<M)-1 down to 0).
            # This is crucial for 0/1 knapsack-like DP where each item (n_val) is used at most once.
            # By iterating downwards, when we calculate new_dp[mask | sub_mask],
            # dp[mask] always refers to the state before considering the current 'n_val'.
            for mask in range((1 << M) - 1, -1, -1):
                if dp[mask] == float('inf'):
                    continue

                # For each 'n_val', we consider which subset of target elements ('sub_mask') it will satisfy.
                # 'sub_mask' can be any non-empty subset of 'target'.
                for sub_mask in range(1, (1 << M)):
                    # Optimization: If all elements in 'sub_mask' are already covered by 'mask',
                    # using 'n_val' for this 'sub_mask' would not cover any *new* target elements.
                    # Since cost is non-negative, this transition cannot improve 'dp[mask]' (it would only be 'dp[mask] + cost').
                    # So, we can skip these redundant calculations.
                    if (mask & sub_mask) == sub_mask: 
                        continue
                    
                    current_lcm = 1
                    # Calculate LCM for the elements corresponding to 'sub_mask'.
                    # Python's integers handle arbitrary size, so intermediate LCM values won't cause overflow.
                    # However, if the LCM becomes excessively large, it implies a very high cost,
                    # making it an unfeasible path towards the optimal solution.
                    for j in range(M):
                        if (sub_mask >> j) & 1: # If j-th target element is in 'sub_mask'
                            t_j = target[j]
                            g = math.gcd(current_lcm, t_j)
                            # Efficient LCM calculation: (a * b) // gcd(a, b) = (a // gcd(a,b)) * b
                            current_lcm = (current_lcm // g) * t_j
                            
                            # Pruning based on max feasible value for an incremented number.
                            # If the LCM itself exceeds this, the resulting needed_val (which would be LCM or a multiple of LCM)
                            # would certainly lead to a cost higher than the maximum possible total operations.
                            if current_lcm > MAX_FEASIBLE_NUM_VALUE:
                                current_lcm = float('inf') # Mark as unfeasible
                                break
                    
                    if current_lcm == float('inf'):
                        continue # This 'sub_mask' requires an unfeasible LCM for 'n_val'
                    
                    # Calculate the smallest multiple of 'current_lcm' that is greater than or equal to 'n_val'.
                    needed_val = ((n_val + current_lcm - 1) // current_lcm) * current_lcm
                    cost = needed_val - n_val

                    # Update dp for the new combined mask (current 'mask' combined with 'sub_mask' covered by 'n_val')
                    new_mask = mask | sub_mask
                    new_dp[new_mask] = min(new_dp[new_mask], dp[mask] + cost)
            
            # After considering all combinations for the current 'n_val', update the main dp array.
            dp = new_dp
            
        # The minimum operations to satisfy all target elements is dp[(1 << M) - 1].
        return dp[(1 << M) - 1]