from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort the start points to handle intervals in order
        starts = sorted(start)
        n = len(starts)
        
        # Helper to check if we can achieve minimum distance `t`
        def can_achieve(t: int) -> bool:
            # Place the first point at the beginning of the first interval
            last = starts[0]
            # For each subsequent interval, we must place its point at least t away
            for i in range(1, n):
                # the earliest we can place for interval i to maintain distance t
                cand = last + t
                # but it must not be before starts[i]
                if cand < starts[i]:
                    cand = starts[i]
                # if this candidate exceeds the end of the interval, fail
                if cand > starts[i] + d:
                    return False
                # accept it and move on
                last = cand
            return True
        
        # Binary search for the maximum t
        # Lower bound is 0, upper bound is the total possible span
        low, high = 0, (starts[-1] + d) - starts[0]
        while low < high:
            mid = (low + high + 1) // 2
            if can_achieve(mid):
                low = mid
            else:
                high = mid - 1
        
        return low