import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def feasible(T):
            sum_x = 0
            for wt in workerTimes:
                if wt <= 0:
                    continue  # According to constraints, wt >=1
                B = T / wt
                if B < 0:
                    x_i = 0
                else:
                    sqrt_val = math.sqrt(1 + 8 * B)
                    x_i = math.floor((-1 + sqrt_val) / 2)
                sum_x += x_i
            return sum_x >= mountainHeight
        
        # Binary search setup
        left = 0
        right = 10**18  # A large enough number to cover maximum possible T
        
        while left < right:
            mid = (left + right) // 2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        return left