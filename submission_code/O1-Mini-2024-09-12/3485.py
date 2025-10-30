from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        intervals = sorted([(s, s + d) for s in start], key=lambda x: x[0])
        n = len(intervals)
        
        def is_possible(min_dist):
            last = -float('inf')
            for interval in intervals:
                low, high = interval
                candidate = max(low, last + min_dist)
                if candidate > high:
                    return False
                last = candidate
            return True
        
        left, right = 0, max(s + d for s in start) - min(s for s in start)
        best = 0
        while left <= right:
            mid = (left + right) // 2
            if is_possible(mid):
                best = mid
                left = mid + 1
            else:
                right = mid - 1
        return best