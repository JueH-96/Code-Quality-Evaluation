from typing import List
import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # Helper to check if we can reduce the mountain in <= T seconds
        def can_finish(T: int) -> bool:
            total = 0
            # For each worker, compute the max x such that t * x*(x+1)/2 <= T
            for t in workerTimes:
                # Solve x*(x+1) <= (2*T)//t
                D = (2 * T) // t
                # x = floor((-1 + sqrt(1 + 4D)) / 2)
                x = (math.isqrt(1 + 4 * D) - 1) // 2
                total += x
                if total >= mountainHeight:
                    return True
            return False

        # Binary search on time T
        # Lower bound 0, upper bound = slowest worker does all the work
        tri = mountainHeight * (mountainHeight + 1) // 2
        lo, hi = 0, max(workerTimes) * tri
        answer = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_finish(mid):
                answer = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return answer