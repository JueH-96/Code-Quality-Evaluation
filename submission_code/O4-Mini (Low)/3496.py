from typing import List
import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # Check if given time T is enough: compute each worker's max capacity
        # under time T, sum them and see if >= mountainHeight.
        def can_finish(T: int) -> bool:
            total = 0
            for t in workerTimes:
                # Solve t * x * (x+1) / 2 <= T
                # => x^2 + x - 2*T/t <= 0
                # root = floor((sqrt(1 + 8*T/t) - 1) / 2)
                # Use float sqrt; double precision suffices for our ranges.
                cap = int((math.isqrt(1 + (8 * T) // t) - 1) // 2)
                # Alternatively, one could use float:
                # cap = int((math.sqrt(1 + 8.0*T/t) - 1) // 2)
                total += cap
                if total >= mountainHeight:
                    return True
            return False

        H = mountainHeight
        # Upper bound: slowest worker does all H units
        max_t = max(workerTimes)
        high = max_t * H * (H + 1) // 2
        low = 0

        # Binary search minimal T
        while low < high:
            mid = (low + high) // 2
            if can_finish(mid):
                high = mid
            else:
                low = mid + 1

        return low