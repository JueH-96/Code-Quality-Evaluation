from typing import List
import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        """
        Binary–search on the answer (makespan).
        For a fixed time `T` every worker i can remove at most
               x where  ti * x * (x + 1) / 2  <=  T
        ⇔  x (x + 1)  <=  2T / ti
        Let  m = ⌊2T / ti⌋ .  The largest integer x that fulfils
               x² + x − m ≤ 0
        is
               x = ⌊ (√(1 + 4m) − 1) / 2 ⌋ .
        If the sum of such capacities over all workers ≥ mountainHeight,
        the mountain can be flattened within T seconds.
        The smallest T that fulfils this predicate is found with binary search.
        """
        
        # Helper: can we finish in at most `time_limit` seconds?
        def feasible(time_limit: int) -> bool:
            capacity = 0
            for t in workerTimes:
                # maximum m = floor(2 * time / t)
                m = (time_limit << 1) // t      # (2 * time_limit) // t, done with shifts for speed
                if m == 0:
                    continue
                # x = floor((sqrt(1 + 4m) - 1) / 2)
                s = math.isqrt(1 + (m << 2))    # isqrt(1 + 4m)
                x = (s - 1) >> 1                # (s - 1) // 2
                capacity += x
                if capacity >= mountainHeight:  # early stop – enough work done
                    return True
            return capacity >= mountainHeight
        
        # Upper bound: let the fastest worker do everything
        fastest = min(workerTimes)
        hi = fastest * mountainHeight * (mountainHeight + 1) // 2
        lo = 0
        
        # Classical binary search for the smallest feasible `hi`
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                hi = mid
            else:
                lo = mid + 1
        
        return lo