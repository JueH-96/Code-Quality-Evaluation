class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        lo, hi = 0, n
        # Binary search on the maximum number of groups k
        while lo < hi:
            mid = (lo + hi + 1) // 2
            # Compute total available usages capped by mid (since each number
            # can appear at most once in each group, and there are mid groups)
            total = 0
            for u in usageLimits:
                total += min(u, mid)
            # Required slots to build groups of size 1,2,...,mid
            required = mid * (mid + 1) // 2
            if total >= required:
                lo = mid
            else:
                hi = mid - 1
        return lo