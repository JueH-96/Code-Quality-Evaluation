import math
from math import gcd
from typing import List

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        k = len(target)
        if k == 0:
            return 0
        full_mask = (1 << k) - 1
        
        lcm_arr = [1] * (full_mask + 1)
        for j in range(k):
            bit = 1 << j
            t_val = target[j]
            for s in range(full_mask + 1):
                if s & bit:
                    current_lcm = lcm_arr[s]
                    new_lcm = current_lcm * t_val // gcd(current_lcm, t_val)
                    lcm_arr[s] = new_lcm
        
        INF = 10**18
        dp = [INF] * (full_mask + 1)
        dp[0] = 0
        
        for num in nums:
            new_dp = dp.copy()
            for mask in range(full_mask + 1):
                if dp[mask] == INF:
                    continue
                u = full_mask ^ mask
                s = u
                while s:
                    L = lcm_arr[s]
                    remainder = num % L
                    cost = 0 if remainder == 0 else L - remainder
                    new_mask = mask | s
                    candidate = dp[mask] + cost
                    if candidate < new_dp[new_mask]:
                        new_dp[new_mask] = candidate
                    s = (s - 1) & u
            dp = new_dp
        
        return dp[full_mask]