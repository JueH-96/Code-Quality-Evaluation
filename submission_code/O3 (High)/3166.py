from typing import List
from collections import Counter
import math

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        """
        Let the minimal size of any group be s.  Then every group has size s or s+1.
        For a value that appears F times we must split F into c groups so that
                 c*s ≤ F ≤ c*(s+1)
        holds for an integer c.  This inequality is feasible iff

                 ceil(F/(s+1)) ≤ c ≤ floor(F/s)

        which in turn is possible iff
                 floor(F/s) ≥ ceil(F/(s+1))

        If it is true for every frequency F, the choice of s is valid.  
        Moreover, for a fixed valid s the fewest groups we can create for that
        frequency is ceil(F/(s+1)), so the total number of groups equals

                 Σ  ceil(F/(s+1))        (taken over all distinct values).

        A larger minimal size s always produces *not more* groups, but the
        feasibility condition becomes harder to satisfy, i.e. the set of valid
        s is downward-closed.  Therefore the overall minimum group count is
        achieved by the largest valid s.

        We can locate this largest s by binary searching over the interval
        [1, min_frequency].  Each feasibility check is O(#distinct values),
        and log₂(min_frequency) ≤ 17 for the given constraints, keeping the
        total time well within limits (≤ 1.7·10⁶ operations).
        """

        freq = list(Counter(nums).values())
        min_freq = min(freq)

        # helper: check if a given minimal size s is feasible
        def feasible(s: int) -> bool:
            for f in freq:
                if f // s < (f + s) // (s + 1):     # floor(f/s) < ceil(f/(s+1))
                    return False
            return True

        # binary search for the largest feasible s
        lo, hi = 1, min_freq
        while lo < hi:
            mid = (lo + hi + 1) // 2      # bias to the right (looking for max)
            if feasible(mid):
                lo = mid
            else:
                hi = mid - 1
        best_s = lo

        # compute the minimal number of groups with this best_s
        ans = sum((f + best_s) // (best_s + 1) for f in freq)
        return ans