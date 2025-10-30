import math
from math import gcd
from typing import List

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        m = len(target)
        total_mask = (1 << m) - 1
        
        L_arr = [0] * (1 << m)
        for mask in range(1, 1 << m):
            val = 1
            for j in range(m):
                if mask & (1 << j):
                    val = self.lcm(val, target[j])
            L_arr[mask] = val
        
        n = len(nums)
        dp = [float('inf')] * (1 << m)
        dp[0] = 0
        
        for i in range(n):
            x = nums[i]
            cost_list = []
            T_prime_list = []
            for submask in range(1, 1 << m):
                L_val = L_arr[submask]
                if x % L_val == 0:
                    y = x
                else:
                    y = (x // L_val + 1) * L_val
                cost_val = y - x
                T_prime = 0
                for j in range(m):
                    if y % target[j] == 0:
                        T_prime |= (1 << j)
                cost_list.append(cost_val)
                T_prime_list.append(T_prime)
            
            ndp = dp.copy()
            for mask in range(1 << m):
                if dp[mask] == float('inf'):
                    continue
                for idx in range(len(cost_list)):
                    cst = cost_list[idx]
                    T_prime_val = T_prime_list[idx]
                    new_mask = mask | T_prime_val
                    if dp[mask] + cst < ndp[new_mask]:
                        ndp[new_mask] = dp[mask] + cst
            dp = ndp
        
        return dp[total_mask]
    
    @staticmethod
    def lcm(a, b):
        return a * b // gcd(a, b)