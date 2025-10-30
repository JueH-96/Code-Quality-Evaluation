from typing import List
import collections

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # We form an interval [nums[i] - k, nums[i] + k] for each i.
        # We want the maximum number of these intervals that overlap at some integer point.
        # Do a sweep-line: +1 at start, -1 after end.
        
        events = collections.Counter()
        
        for x in nums:
            start = x - k
            end = x + k
            events[start] += 1
            # We subtract at end+1 so that the interval [start, end] is inclusive
            events[end + 1] -= 1
        
        max_beauty = 0
        cur = 0
        # Sweep through all event points in sorted order
        for point in sorted(events):
            cur += events[point]
            if cur > max_beauty:
                max_beauty = cur
        
        return max_beauty