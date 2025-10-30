from typing import List
import math

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        m = len(target)
        full_mask = (1 << m) - 1
        
        # Precompute the LCM for each non-empty subset mask of targets
        lcm_of_mask = [0] * (1 << m)
        for mask in range(1, 1 << m):
            l = 1
            for i in range(m):
                if mask & (1 << i):
                    l = l * target[i] // math.gcd(l, target[i])
            lcm_of_mask[mask] = l
        
        # For each mask, find the minimal cost across all nums elements
        # to reach a multiple of lcm_of_mask[mask]
        INF = 10**18
        best_cost = [INF] * (1 << m)
        for a in nums:
            # for each mask
            for mask in range(1, 1 << m):
                l = lcm_of_mask[mask]
                # cost to bump a up to next multiple of l
                # ceil(a / l) * l - a
                # if a % l == 0 => cost = 0
                rem = a % l
                if rem == 0:
                    cost = 0
                else:
                    cost = l - rem
                if cost < best_cost[mask]:
                    best_cost[mask] = cost
        
        # DP over mask subsets to cover all targets
        dp = [INF] * (1 << m)
        dp[0] = 0
        for cur in range(1 << m):
            if dp[cur] == INF:
                continue
            # try adding one block that covers 'mask2'
            for mask2 in range(1, 1 << m):
                new_mask = cur | mask2
                new_cost = dp[cur] + best_cost[mask2]
                if new_cost < dp[new_mask]:
                    dp[new_mask] = new_cost
        
        return dp[full_mask]