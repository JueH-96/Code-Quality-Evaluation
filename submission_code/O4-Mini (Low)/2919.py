from bisect import bisect_right
from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usage limits
        usageLimits.sort()
        n = len(usageLimits)
        
        # Compute prefix sums for quick range sum queries
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + usageLimits[i]
        
        # Function to check if we can form k groups of sizes 1, 2, ..., k
        def can_make(k: int) -> bool:
            # Total items needed = k*(k+1)//2
            need = k * (k + 1) // 2
            # Find how many limits are <= k
            idx = bisect_right(usageLimits, k)
            # Sum of min(limit, k) = sum( limits[0:idx] ) + k * (n - idx)
            total_available = prefix[idx] + k * (n - idx)
            return total_available >= need
        
        # Binary search on the number of groups k
        # Lower bound 0, upper bound such that k*(k+1)/2 <= total sum of usageLimits
        total_sum = prefix[n]
        lo, hi = 0, int((2 * total_sum) ** 0.5) + 2  # +2 for safety margin
        
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can_make(mid):
                lo = mid
            else:
                hi = mid - 1
        
        return lo