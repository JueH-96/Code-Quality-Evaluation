from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # We transform each nums[i] into an interval [nums[i] - k, nums[i] + k]
        # Then we want the point v that lies in the maximum number of intervals.
        # This is a classic "maximum interval overlap" problem, solvable by sweep line.
        
        events = {}
        for x in nums:
            start = x - k
            end = x + k
            # At the start point, one interval begins
            events[start] = events.get(start, 0) + 1
            # Just past the end point, one interval ends
            # We use end + 1 because intervals are inclusive
            events[end + 1] = events.get(end + 1, 0) - 1
        
        # Sweep through the events in order of coordinate
        current = 0
        max_beauty = 0
        for coord in sorted(events):
            current += events[coord]
            if current > max_beauty:
                max_beauty = current
        
        return max_beauty