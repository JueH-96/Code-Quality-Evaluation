import math
from typing import List

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        m = len(target)
        if m == 0:
            return 0
        
        targets = target
        # Precompute LCM for all non-empty subsets of targets
        lcm_for_subset = [0] * (1 << m)
        for mask in range(1, 1 << m):
            current_targets = []
            for i in range(m):
                if (mask & (1 << i)):
                    current_targets.append(targets[i])
            current_lcm = 1
            for t in current_targets:
                current_lcm = current_lcm * t // math.gcd(current_lcm, t)
            lcm_for_subset[mask] = current_lcm
        
        INF = float('inf')
        dp = [INF] * (1 << m)
        dp[0] = 0  # Base case: no targets covered with 0 operations
        
        for num in nums:
            temp_dp = dp.copy()
            # Iterate over all non-empty subsets
            for mask in range(1, 1 << m):
                lcm_val = lcm_for_subset[mask]
                required_multiple = lcm_val
                # Compute delta to reach the next multiple of required_multiple
                delta = (( (num + required_multiple - 1) // required_multiple ) * required_multiple) - num
                
                # Update temp_dp based on this subset's contribution
                for current_mask in range(0, 1 << m):
                    if dp[current_mask] == INF:
                        continue
                    new_mask = current_mask | mask
                    new_cost = dp[current_mask] + delta
                    if new_cost < temp_dp[new_mask]:
                        temp_dp[new_mask] = new_cost
            # Update dp to the new state after considering this num
            dp = temp_dp
        
        full_mask = (1 << m) - 1
        return dp[full_mask] if dp[full_mask] != INF else -1