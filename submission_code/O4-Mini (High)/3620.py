from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Each element num can be moved anywhere in [num - k, num + k].
        # We want to pick as many distinct integer points as possible,
        # one from each interval, so that no two picked points coincide.
        # This is a classic "interval scheduling" on integers:
        # sort intervals by right endpoint, then greedily assign
        # the smallest available integer in each interval.
        
        # Build intervals [l, r] for each num
        intervals = [(num - k, num + k) for num in nums]
        # Sort by right endpoint
        intervals.sort(key=lambda x: x[1])
        
        res = 0
        # current assigned point (last integer we used)
        # initialize to a very small number
        curr = -10**18
        
        for l, r in intervals:
            # If we haven't reached the interval yet, place at its left end
            if curr < l:
                curr = l
                res += 1
            # Otherwise, try to place at curr+1 if it still lies in [l, r]
            elif curr + 1 <= r:
                curr += 1
                res += 1
            # else: no room to place a new distinct integer in this interval
        
        return res