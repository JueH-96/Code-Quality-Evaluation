import bisect
from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        sorted_L = sorted(usageLimits)
        n = len(sorted_L)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + sorted_L[i]
        
        low = 0
        high = 2 * 10**18  # A sufficiently large upper bound
        best = 0
        while low <= high:
            mid = (low + high) // 2
            # Find the first index where L_i >= mid
            pos = bisect.bisect_left(sorted_L, mid)
            sum_min = prefix[pos] + (n - pos) * mid
            required = mid * (mid + 1) // 2
            if sum_min >= required:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best