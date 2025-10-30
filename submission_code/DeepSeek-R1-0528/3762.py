import math

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        lo = 0
        hi = 10**18
        
        while lo < hi:
            mid = (lo + hi + 1) // 2
            required = [0] * n
            for i in range(n):
                if mid == 0:
                    required[i] = 0
                else:
                    required[i] = (mid + points[i] - 1) // points[i]
            
            extra = 0
            for i in range(n):
                if required[i] > 1:
                    extra = max(extra, (required[i] - 1) * 2 * min(i, n - 1 - i))
            
            moves_needed = n - 1 + extra
            if 1 + moves_needed <= m:
                lo = mid
            else:
                hi = mid - 1
        
        return lo