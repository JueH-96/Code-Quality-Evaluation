from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort the intervals by their starting point.
        start.sort()
        n = len(start)
        
        # Helper function: given a candidate minimum gap "m", check if one can select
        # one integer from each interval [start[i], start[i] + d] so that when the chosen
        # numbers are sorted, consecutive numbers differ by at least m.
        def can_place(m: int) -> bool:
            # For the first interval, we choose the smallest possible value.
            current = start[0]  
            for i in range(1, n):
                # In the i-th interval [start[i], start[i]+d], we need a number at least m
                # larger than the last chosen number 'current'.  To keep options open for future choices,
                # we choose the smallest number in the interval that is >= current + m.
                candidate = max(current + m, start[i])
                # If candidate is greater than the right bound of the interval, then it's not possible.
                if candidate > start[i] + d:
                    return False
                current = candidate
            return True
        
        # Binary search for the maximum possible integer m (the score).
        # Lower bound for m is 0. 
        # The highest m is bounded by the overall span: 
        # the farthest right possible number (max(start) + d) minus the smallest possible number (min(start)).
        lo, hi = 0, start[-1] + d - start[0]
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_place(mid):
                ans = mid
                lo = mid + 1  # try a larger gap
            else:
                hi = mid - 1  # reduce the gap candidate
        return ans