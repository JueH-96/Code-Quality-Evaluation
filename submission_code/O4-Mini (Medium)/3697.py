from typing import List
import math

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        m = len(target)
        # Precompute lcm for every non-empty subset of targets (mask from 1 to (1<<m)-1)
        max_mask = 1 << m
        lcms = [0] * max_mask
        # For singleton masks, lcm is the target itself
        for i in range(m):
            lcms[1 << i] = target[i]
        # For larger masks, build from submasks
        for mask in range(1, max_mask):
            if lcms[mask] == 0:
                # split off lowest set bit
                low = mask & -mask
                rest = mask ^ low
                lcms[mask] = (lcms[low] * lcms[rest]) // math.gcd(lcms[low], lcms[rest])
        
        FULL = max_mask - 1
        INF = 10**18
        # dp[mask] = min cost to cover targets in 'mask'
        dp = [INF] * max_mask
        dp[0] = 0
        
        for k in nums:
            # Precompute cost for this element to cover each subset
            cost = [0] * max_mask
            for mask in range(1, max_mask):
                l = lcms[mask]
                # ops to raise k to next multiple of l
                r = k % l
                cost[mask] = (l - r) % l
            
            # 0/1 style DP update: either skip this element or use it once
            dp2 = dp[:]  # copy
            for old_mask in range(max_mask):
                base = dp[old_mask]
                if base == INF:
                    continue
                # try covering any non-empty subset S with this element
                for s in range(1, max_mask):
                    new_mask = old_mask | s
                    c = base + cost[s]
                    if c < dp2[new_mask]:
                        dp2[new_mask] = c
            dp = dp2
        
        return dp[FULL]