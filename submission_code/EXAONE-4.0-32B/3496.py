import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        if mountainHeight == 0:
            return 0
        
        max_worker = max(workerTimes)
        high_T = max_worker * (mountainHeight * (mountainHeight + 1)) // 2
        low_T = 0
        
        while low_T < high_T:
            mid_T = (low_T + high_T) // 2
            total_cap = 0
            for wt in workerTimes:
                if mid_T == 0:
                    cap = 0
                else:
                    D = 1 + (8 * mid_T) // wt
                    sqrt_D = math.isqrt(D)
                    cap = (sqrt_D - 1) // 2
                total_cap += cap
                if total_cap >= mountainHeight:
                    break
            
            if total_cap >= mountainHeight:
                high_T = mid_T
            else:
                low_T = mid_T + 1
        
        return low_T