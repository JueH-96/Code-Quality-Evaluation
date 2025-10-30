import math
from typing import List

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        
        unique_target = sorted(list(set(target)))
        k = len(unique_target)

        def lcm(a, b):
            if a == 1: return b
            if b == 1: return a
            return (a * b) // math.gcd(a, b)
        
        # 1. Precompute LCM for all subsets of target
        lcm_map = [0] * (1 << k)
        lcm_map[0] = 1
        for i in range(k):
            ti = unique_target[i]
            for mask_val in range(1 << i):
                prev_lcm = lcm_map[mask_val]
                new_mask = mask_val | (1 << i)
                lcm_map[new_mask] = lcm(prev_lcm, ti)

        # 2. Precompute min cost to satisfy each sub-problem with one number
        cost_for_submask = [0] * (1 << k)
        for mask in range(1, 1 << k):
            L = lcm_map[mask]
            min_cost_for_L = float('inf')
            for n in nums:
                # Cost to make n a multiple of L
                cost = (L - (n % L)) % L
                min_cost_for_L = min(min_cost_for_L, cost)
            cost_for_submask[mask] = min_cost_for_L
            
        # 3. DP to find min cost for covering all targets
        dp = [float('inf')] * (1 << k)
        dp[0] = 0
        
        for mask in range(1, 1 << k):
            # Iterate through all non-empty submasks of the current mask
            submask = mask
            while submask > 0:
                prev_mask = mask ^ submask
                current_cost = dp[prev_mask] + cost_for_submask[submask]
                dp[mask] = min(dp[mask], current_cost)
                
                # Move to the next submask
                submask = (submask - 1) & mask

        return dp[(1 << k) - 1]