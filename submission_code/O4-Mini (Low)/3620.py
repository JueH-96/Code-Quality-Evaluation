from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Each number nums[i] defines an interval [nums[i]-k, nums[i]+k].
        # We need to choose a distinct integer within each interval,
        # maximizing the count of distinct picks.  This reduces to
        # finding a maximum matching between n intervals and integer points.
        # A classic greedy method: sort intervals by their right endpoint,
        # then for each interval pick the smallest available integer ≥ its left
        # endpoint that is greater than the last assigned integer.
        
        intervals = []
        for x in nums:
            intervals.append((x - k, x + k))
        
        # Sort by right endpoint
        intervals.sort(key=lambda iv: iv[1])
        
        last_assigned = -10**30  # something smaller than any possible l
        count = 0
        
        for l, r in intervals:
            # the earliest integer we can assign is max(l, last_assigned+1)
            candidate = max(l, last_assigned + 1)
            if candidate <= r:
                # we can assign this candidate
                last_assigned = candidate
                count += 1
        
        return count