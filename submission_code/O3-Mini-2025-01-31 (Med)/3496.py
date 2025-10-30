from typing import List
import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # Helper function that checks if in T seconds, the workers can reduce the mountain by at least mountainHeight
        def canComplete(T: int) -> bool:
            total = 0
            # For each worker, compute the maximum number of height reductions (x) that can be done in T seconds.
            for t in workerTimes:
                # The formula for the time needed to perform x reductions is:
                #    time = t * (1 + 2 + ... + x) = t * x * (x+1) // 2
                # We need t * x*(x+1)//2 <= T  -> x*(x+1) <= 2*T/t
                # Let limit = floor(2*T / t)
                limit = (2 * T) // t
                # Solve x*(x+1) <= limit
                # This is a quadratic inequality: x^2 + x - limit <= 0.
                # The positive root of equality is: x = (-1 + sqrt(1 + 4*limit)) / 2.
                x = int((math.sqrt(1 + 4 * limit) - 1) // 2)
                total += x
                # Early exit if already enough height reductions are achieved.
                if total >= mountainHeight:
                    return True
            return total >= mountainHeight

        # Binary search on the time T.
        # The worst-case time can be estimated by when only one worker works.
        # For a worker with time t, reducing mountain by mountainHeight will take time:
        #    t * (mountainHeight * (mountainHeight + 1) // 2)
        # Over all workers, the fastest worker is the one with min(workerTimes)
        t_min = min(workerTimes)
        hi = t_min * mountainHeight * (mountainHeight + 1) // 2
        lo = 0
        ans = hi  # initialize answer with worst-case scenario

        while lo <= hi:
            mid = (lo + hi) // 2
            if canComplete(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1

        return ans