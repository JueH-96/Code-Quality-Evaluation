from typing import List
import math

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        n = len(nums)
        # Count frequencies for each distinct value.
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        counts = list(freq.values())
        # Lower bound on m: at least one group for each distinct value.
        low = len(counts)
        high = n  # worst-case: each index in its own group
        
        # Check feasibility for a given m.
        def valid(m: int) -> bool:
            L = n // m       # minimum group size
            r = n - m * L    # number of groups that must have size L+1
            # Every group must have at least L entries.
            # So for every frequency f, we need f >= L.
            for f in counts:
                if f < L:
                    return False
            
            if r == 0:
                # All groups are exactly size L.
                # Then each value v must partition its f completely into groups of size L,
                # so f must be divisible by L and k_v = f // L.
                total_groups = 0
                for f in counts:
                    if f % L != 0:
                        return False
                    total_groups += f // L
                return total_groups == m
            else:
                total_min = 0  # minimum groups we must use overall
                total_max = 0  # maximum groups we could possibly subdivide into
                for f in counts:
                    # Minimum groups required if we want to use groups of size at most L+1:
                    kmin = math.ceil(f / (L + 1))
                    # Maximum groups possible if groups have at least L entries:
                    kmax = f // L
                    total_min += kmin
                    total_max += kmax
                # We have flexibility if the overall assignment can use between total_min and total_max groups.
                return total_min <= m <= total_max

        # Binary search for minimal m that is valid.
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if valid(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans