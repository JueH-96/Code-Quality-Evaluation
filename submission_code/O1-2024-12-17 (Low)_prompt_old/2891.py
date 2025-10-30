from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        """
        We want the longest subsequence of equal elements after possibly
        adjusting each element nums[i] once (or not at all) into any value
        within [nums[i]-k, nums[i]+k]. This is equivalent to finding a single
        integer X that lies in the maximum number of intervals 
        [nums[i]-k, nums[i]+k], because each chosen index i can be turned into X
        if and only if X is in that interval.

        Therefore, we construct intervals for each element and find the maximum
        number of intervals that overlap at any point. We use a standard line-
        sweep (or "sweep line") approach:

        1. For each element nums[i], form the interval [start, end] = [nums[i]-k, nums[i]+k].
        2. Collect "events": (start, +1) when an interval starts, (end, -1) when an interval ends.
           (These ends are inclusive, so we will process end-events before start-events if they
            share the same coordinate to avoid counting an interval that ends exactly at a point
            and another that starts at the same point as overlapping.)
        3. Sort events by coordinate; if coordinates tie, sort by delta (so that -1 is processed
           before +1).
        4. Sweep from left to right, accumulating a "coverage" which tracks how many intervals
           cover the current point. The maximum coverage encountered is the answer.
        """
        # Build the list of events
        events = []
        for num in nums:
            start = num - k
            end = num + k
            # Add start and end events (end is inclusive, so we use (end, -1)).
            events.append((start, +1))
            events.append((end, -1))
        
        # Sort the events:
        #   - by coordinate ascending
        #   - if tie, process -1 (end) before +1 (start)
        events.sort(key=lambda x: (x[0], x[1]))
        
        coverage = 0   # How many intervals cover the current point
        max_coverage = 0
        for _, delta in events:
            coverage += delta
            max_coverage = max(max_coverage, coverage)
        
        return max_coverage