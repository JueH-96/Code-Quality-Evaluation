from typing import List
import math

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # Helper function: given T, how many height units can be reduced in total.
        def total_reduction(T: int) -> int:
            total = 0
            for t in workerTimes:
                # Solve for maximum integer x such that t * (x*(x+1)//2) <= T.
                # That is, x*(x+1) <= 2*T/t.
                # We solve quadratic equation: x^2 + x - (2*T/t) <= 0.
                # The positive root is (-1 + sqrt(1 + 8*T/t)) // 2.
                # Use float division for sqrt.
                possible = int((-1 + math.sqrt(1 + 8 * T / t)) // 2)
                total += possible
                if total >= mountainHeight:
                    return total
            return total
        
        # Binary search for the minimal T to achieve at least mountainHeight reduction.
        # Lower bound (lo) can be 0 seconds.
        lo, hi = 0, min(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        
        while lo < hi:
            mid = (lo + hi) // 2
            if total_reduction(mid) >= mountainHeight:
                hi = mid
            else:
                lo = mid + 1
        return lo