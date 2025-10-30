import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        workerTimes.sort()  # Sort worker times in ascending order
        
        def is_possible(T):
            total = 0
            for t in workerTimes:
                if T < t:
                    continue
                # Find the maximum x such that t * x*(x+1)//2 <= T
                low, high = 0, 1
                while t * high * (high + 1) // 2 <= T:
                    low = high
                    high *= 2
                best_x = 0
                while low <= high:
                    mid = (low + high) // 2
                    current = t * mid * (mid + 1) // 2
                    if current <= T:
                        best_x = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                total += best_x
                if total >= mountainHeight:
                    return True
            return total >= mountainHeight
        
        # Determine the upper bound for binary search
        t_max = max(workerTimes)
        left = 0
        right = t_max * mountainHeight * (mountainHeight + 1)
        
        # Perform binary search to find the minimal feasible time
        while left < right:
            mid = (left + right) // 2
            if is_possible(mid):
                right = mid
            else:
                left = mid + 1
        return left