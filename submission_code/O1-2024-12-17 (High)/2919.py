from typing import List
from bisect import bisect_right
from math import isqrt

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usage limits
        usageLimits.sort()
        n = len(usageLimits)

        # Build prefix sums for quick sum( min(usageLimits[i], k) ) calculation
        prefix = [0] * n
        prefix[0] = usageLimits[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] + usageLimits[i]

        # Total usage possible (sum of all usageLimits)
        total = prefix[-1]

        # Helper to compute sum of min(usageLimits[i], k) in O(log n)
        # using the sorted array and prefix sums
        def sum_min_k(k: int) -> int:
            # Find how many elements are <= k
            idx = bisect_right(usageLimits, k)
            # Sum of all elements <= k
            s = prefix[idx-1] if idx > 0 else 0
            # For the remaining elements (which are > k), each contributes k
            s += (n - idx) * k
            return s

        # The maximum possible number of groups cannot exceed the largest k
        # satisfying k*(k+1)/2 <= total.
        # Solve k*(k+1)/2 <= total via quadratic formula bound:
        max_k = int((isqrt(1 + 8 * total) - 1) // 2)

        # Binary search for the largest k such that sum_min_k(k) >= k*(k+1)//2
        low, high = 0, max_k
        while low < high:
            mid = (low + high + 1) // 2
            if sum_min_k(mid) >= mid * (mid + 1) // 2:
                low = mid
            else:
                high = mid - 1

        return low