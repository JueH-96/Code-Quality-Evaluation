import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        low = 0
        max_t = max(workerTimes)
        high = (mountainHeight * (mountainHeight + 1) // 2) * max_t
        while low < high:
            mid = (low + high) // 2
            total = 0
            for t in workerTimes:
                if mid == 0:
                    x = 0
                else:
                    val = 8 * mid / t
                    sqrt_val = math.sqrt(1 + val)
                    x = int((sqrt_val - 1) / 2)
                    # Check if x+1 is possible
                    while (x + 1) * (x + 2) * t <= 2 * mid:
                        x += 1
                total += x
            if total >= mountainHeight:
                high = mid
            else:
                low = mid + 1
        return high