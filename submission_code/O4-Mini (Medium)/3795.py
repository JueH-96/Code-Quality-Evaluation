from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        # Precompute for each index j the list of (query_index, val) that affect j
        aff = [[] for _ in range(n)]
        for i, (l, r, v) in enumerate(queries):
            for j in range(l, r+1):
                aff[j].append((i, v))
        # The maximum target value we need to reach is max(nums)
        max_t = max(nums) if nums else 0
        # A bitmask to trim dp to only bits 0..max_t
        mask = (1 << (max_t + 1)) - 1

        # Check if using the first k queries we can zero out the array
        def can_zero(k: int) -> bool:
            # If k == 0, we need all nums already zero
            if k == 0:
                return all(x == 0 for x in nums)
            # For each position j, do subset-sum DP with the values of queries < k that cover j
            for j in range(n):
                target = nums[j]
                # dp is a bitset: bit s is 1 if sum s is achievable
                dp = 1  # only sum 0 achievable at start
                for qi, val in aff[j]:
                    if qi >= k:
                        break
                    # include this val in the subset-sum
                    dp |= (dp << val)
                    # mask off bits above max_t, we only need up to max_t
                    dp &= mask
                    # early exit: if we've already reached target, we can stop
                    if (dp >> target) & 1:
                        break
                # if target not achievable, fail
                if not ((dp >> target) & 1):
                    return False
            return True

        # Binary search the minimal k in [0..m] for which can_zero(k) is True
        lo, hi = 0, m
        while lo < hi:
            mid = (lo + hi) // 2
            if can_zero(mid):
                hi = mid
            else:
                lo = mid + 1

        # Verify final answer
        if lo <= m and can_zero(lo):
            return lo
        return -1