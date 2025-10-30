from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for x in nums:
            start = x - k
            end = x + k
            events.append((start, 1))   # Start event
            events.append((end + 1, -1)) # End event
        
        # Sort events by x coordinate; if same x, end events (-1) come before start events (1)
        events.sort(key=lambda e: (e[0], e[1]))
        
        current_coverage = 0
        max_coverage = 0
        
        for x, delta in events:
            current_coverage += delta
            if current_coverage > max_coverage:
                max_coverage = current_coverage
        
        return max_coverage