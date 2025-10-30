import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        if mountainHeight == 0:
            return 0
        
        max_wt = max(workerTimes)
        high = max_wt * (mountainHeight * (mountainHeight + 1) // 2)
        low = 0
        
        def feasible(T):
            total_units = 0
            for wt in workerTimes:
                if T < wt:
                    x = 0
                else:
                    discriminant = 1 + (8 * T) / wt
                    x_val = (-1 + math.sqrt(discriminant)) / 2
                    x_floor = math.floor(x_val)
                    if (x_floor + 1) * (x_floor + 2) * wt <= 2 * T:
                        x = x_floor + 1
                    else:
                        x = x_floor
                total_units += x
                if total_units >= mountainHeight:
                    break
            return total_units >= mountainHeight
        
        while low < high:
            mid = (low + high) // 2
            if feasible(mid):
                high = mid
            else:
                low = mid + 1
        return low