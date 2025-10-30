from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Generate and sort intervals by their start
        intervals = sorted((s, s + d) for s in start)
        n = len(intervals)
        
        # Helper to check if we can place points with at least 'dist' apart
        def can_achieve(dist: int) -> bool:
            # place the first point at the earliest possible
            prev = intervals[0][0]
            for i in range(1, n):
                lo, hi = intervals[i]
                # the next point must be at least prev + dist
                needed = prev + dist
                if needed > hi:
                    return False
                # we place it at the max of lo and needed
                prev = max(lo, needed)
            return True
        
        # Binary search for the maximum feasible minimum distance
        lo, hi = 0, intervals[-1][1] - intervals[0][0]  # possible range of distances
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_achieve(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        return ans