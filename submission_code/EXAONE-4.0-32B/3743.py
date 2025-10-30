import bisect
from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        total_duration = 0
        for i in range(n):
            total_duration += endTime[i] - startTime[i]
        F = eventTime - total_duration
        if F == 0:
            return 0
        
        def Check(X):
            i0 = bisect.bisect_left(startTime, X)
            if i0 <= k:
                return True
            
            pos = bisect.bisect_right(endTime, eventTime - X)
            i0 = pos - 1
            if i0 >= 0 and (n - 1 - i0) <= k:
                return True
            
            for i in range(n - 1):
                low_bound = endTime[i] + X
                lo, hi = i + 1, n - 1
                j_found = n
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if startTime[mid] >= low_bound:
                        j_found = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                if j_found < n and (j_found - i - 1) <= k:
                    return True
            
            return False
        
        if Check(F):
            return F
        else:
            low, high = 0, F
            ans = 0
            while low <= high:
                mid = (low + high) // 2
                if Check(mid):
                    ans = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return ans