from typing import List
import math

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        n = len(nums)
        # frequency map
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1

        # function to check if a given k (groups) is feasible.
        def can_assign(k: int) -> bool:
            # k groups partition the n indices:
            q, r = divmod(n, k)
            total_groups_used = 0
            if r == 0:
                # Every group must have exactly group_size = n/k = q.
                # Then for each distinct number v with frequency f,
                # we must have f divisible by group_size.
                for f in freq.values():
                    if f % q != 0:
                        return False
                    # v will use exactly f/(group_size) groups.
                    total_groups_used += f // q
                return total_groups_used == k
            else:
                # In this case groups sizes are either q or q+1.
                # For each distinct value v with frequency f, a natural split is:
                # use g = ceil(f/(q+1)) groups.
                # Then if f mod g equals 0, every part is f/g (which must equal q+1)
                # Else the nearly–equal split produces parts f//g and f//g+1.
                # In order to be valid, we must have that these numbers are exactly q and q+1.
                local_sum = 0
                for f in freq.values():
                    g = math.ceil(f / (q+1))
                    # For validity, we need the “small part” be exactly q.
                    # When f is split into g parts, the “small part” is f//g.
                    if f // g != q:
                        return False
                    local_sum += g
                return local_sum == k

        # Binary search on k in the range [1, n].
        lo, hi = 1, n
        ans = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_assign(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans