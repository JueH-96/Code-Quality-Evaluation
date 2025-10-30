from typing import List
import math

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count frequencies of each value
        freq = {}
        for x in nums:
            freq[x] = freq.get(x, 0) + 1
        counts = list(freq.values())
        n = len(nums)

        # Helper: can we partition into G groups?
        def can(G: int) -> bool:
            # k = floor(n/G): minimal group size
            # we allow group sizes k or k+1
            k = n // G
            # Compute minimal and maximal groups needed across all values
            lo_sum = 0
            hi_sum = 0
            for c in counts:
                # minimal groups if we use as many size-(k+1) groups as possible
                lo = math.ceil(c / (k + 1))
                lo_sum += lo
                # maximal groups if we use as many size-k groups as possible
                # if k == 0, we cannot form any size-0 groups, so hi = infinite
                if k > 0:
                    hi = c // k
                else:
                    hi = n  # large enough
                hi_sum += hi
                # early exit
                if lo_sum > G:
                    return False
            # We need sum(lo) <= G <= sum(hi)
            return lo_sum <= G <= hi_sum

        # Binary search for minimum G in [1..n]
        left, right = 1, n
        ans = n
        while left <= right:
            mid = (left + right) // 2
            if can(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans