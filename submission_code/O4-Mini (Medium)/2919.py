from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        # Binary search on the number of groups k
        # Check if we can form k groups of sizes 1,2,...,k
        # Total needed = k*(k+1)//2
        # Total available = sum(min(limit, k) for each limit)
        # because each number can appear at most once per group, and at most k groups.
        left, right = 0, n
        while left < right:
            mid = (left + right + 1) // 2
            need = mid * (mid + 1) // 2
            # compute how many elements we can contribute if we try to build mid groups
            have = 0
            for limit in usageLimits:
                # each number can contribute at most once per group and at most usageLimits[i] in total
                have += min(limit, mid)
                # early exit if we've already reached need
                if have >= need:
                    break
            if have >= need:
                left = mid
            else:
                right = mid - 1
        return left