from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort(reverse=True)
        n = len(usageLimits)
        sum_limits = sum(usageLimits)
        m1 = min(usageLimits[i] + i for i in range(n))
        # Compute m2 using quadratic formula
        m2 = int((( (8 * sum_limits + 1) ** 0.5 ) - 1) // 2)
        return min(m1, m2, n)