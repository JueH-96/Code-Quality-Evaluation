import math
from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        workerTimes.sort()
        max_t = max(workerTimes)
        low = 0
        high = max_t * mountainHeight * (mountainHeight + 1) // 2  # Initial upper bound
        
        answer = high  # Initialize with the highest possible value
        
        while low <= high:
            mid = (low + high) // 2
            total = 0
            for t in workerTimes:
                # Calculate the maximum x for current worker and time mid
                s = 1 + 8 * mid // t
                x = (math.isqrt(s) - 1) // 2
                total += x
                if total >= mountainHeight:
                    break  # Early exit if sum exceeds the required height
            if total >= mountainHeight:
                answer = mid
                high = mid - 1  # Try to find a smaller feasible T
            else:
                low = mid + 1  # Increase lower bound
        return answer