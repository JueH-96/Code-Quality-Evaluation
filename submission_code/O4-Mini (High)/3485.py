from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort interval start points
        start.sort()
        # Compute the overall span
        min_start = start[0]
        max_end = start[-1] + d

        # Check if it's possible to pick one point per interval
        # so that the minimum pairwise distance is at least T.
        def feasible(T: int) -> bool:
            last = -10**18  # a very small number, so first pick is just start[0]
            for s in start:
                # Earliest we can pick in this interval is max(s, last+T)
                pick = max(s, last + T)
                # If that exceeds the interval's end, T is not feasible
                if pick > s + d:
                    return False
                last = pick
            return True

        # Binary search on the answer T in [0, max_end-min_start]
        lo, hi = 0, max_end - min_start
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if feasible(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo