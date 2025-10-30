import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        if not workerTimes:
            return 0
        
        # Compute the initial high value as the minimal time if any worker does all the work alone
        min_Ti = float('inf')
        for t in workerTimes:
            Ti = (mountainHeight * (mountainHeight + 1) // 2) * t
            if Ti < min_Ti:
                min_Ti = Ti
        high = min_Ti
        low = 0

        def compute_x(T: int, t_i: int) -> int:
            if T == 0:
                return 0
            temp = (2 * T) // t_i
            if temp == 0:
                return 0
            low_x = 0
            high_x = int(math.isqrt(temp)) + 1
            best = 0
            while low_x <= high_x:
                mid_x = (low_x + high_x) // 2
                product = mid_x * (mid_x + 1)
                if product <= temp:
                    best = mid_x
                    low_x = mid_x + 1
                else:
                    high_x = mid_x - 1
            return best

        while low < high:
            mid = (low + high) // 2
            total = 0
            for t in workerTimes:
                x = compute_x(mid, t)
                total += x
                if total >= mountainHeight:
                    break  # Early exit if the total is already sufficient
            if total >= mountainHeight:
                high = mid
            else:
                low = mid + 1
        return high