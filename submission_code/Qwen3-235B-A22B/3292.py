from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        required = set(range(1, n + 1))
        available = set(changeIndices)
        if not required.issubset(available):
            return -1

        def can(s: int) -> bool:
            if s == 0:
                return False
            last_occ = [-1] * (n + 1)  # 1-based indexing for indices
            for t in range(s):
                idx = changeIndices[t]
                last_occ[idx] = t + 1  # store 1-based time
            # Check all indices have been covered
            for i in range(1, n + 1):
                if last_occ[i] == -1:
                    return False
            # Collect pairs of (s_i, nums[i])
            pairs = []
            for i in range(1, n + 1):
                pairs.append((last_occ[i], nums[i - 1]))
            # Sort based on s_i
            pairs.sort()
            sorted_s_i = [x[0] for x in pairs]
            sorted_nums = [x[1] for x in pairs]
            # Compute prefix sums
            prefix = [0] * (n + 1)
            for i in range(n):
                prefix[i + 1] = prefix[i] + sorted_nums[i]
            # Check constraints
            for k in range(n):
                required_time = prefix[k + 1] + (k + 1)
                if required_time > sorted_s_i[k]:
                    return False
            return True

        # Binary search between 1 and m
        lo, hi = 1, m
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans