import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # Binary search on the time T
        # Check if with time T, the workers can collectively reduce at least mountainHeight
        def can_finish(T: int) -> bool:
            total = 0
            for t in workerTimes:
                # Each worker can do at most h such that t * h*(h+1)/2 <= T
                # => h*(h+1) <= 2*T/t
                d = (2 * T) // t
                # solve h^2 + h - d <= 0  =>  h <= (sqrt(1+4d)-1)/2
                s = math.isqrt(1 + 4 * d)
                h = (s - 1) // 2
                total += h
                if total >= mountainHeight:
                    return True
            return False

        # If there's only one worker, he must do all the work
        # Upper bound: fastest worker doing all mountainHeight
        t_min = min(workerTimes)
        high = t_min * mountainHeight * (mountainHeight + 1) // 2
        low = 0

        # Standard binary search for the minimal feasible T
        while low < high:
            mid = (low + high) // 2
            if can_finish(mid):
                high = mid
            else:
                low = mid + 1
        return low