from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # The observation is: we want to form m groups with sizes 1, 2, …, m.
        # In total, we need m*(m+1)//2 element uses. However, we cannot exceed
        # usageLimits[i] for each number i and each number i can be used at most once per group.
        #
        # A key insight is that across all numbers, the total available contribution “at most m per number”
        # is sum(min(usageLimits[i], m)) because even if usageLimits[i] is high, number i can only contribute once per group.
        # Thus, a necessary and sufficient condition to be able to form m groups is:
        #    sum(min(usageLimits[i], m)) >= m*(m+1)//2
        #
        # We can binary search for the maximum m satisfying this condition.
        
        # Compute an upper bound for m.
        total = sum(usageLimits)
        # m*(m+1)//2 <= total --> m ~ O(sqrt(2*total)).
        # Alternatively, an upper bound is m_max = int((sqrt(8*total+1)-1)//2).
        # However, we can search in [0, m_max]
        import math
        m_max = int((math.sqrt(8 * total + 1) - 1) // 2)
        
        def can_form(m: int) -> bool:
            # Check if we can form m groups given usageLimits.
            # For each number, its max contribution is min(limit, m) (cannot be used more than once per group)
            # Sum these contributions and compare with required total uses m*(m+1)//2.
            needed = m * (m + 1) // 2
            total_contribution = 0
            for limit in usageLimits:
                # Each number can help in at most m groups, but only if usageLimits[i] >= that.
                total_contribution += min(limit, m)
                # Early exit if we've already reached the needed total.
                if total_contribution >= needed:
                    return True
            return total_contribution >= needed
        
        lo, hi = 0, m_max
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_form(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans