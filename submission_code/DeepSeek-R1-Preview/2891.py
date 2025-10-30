from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for x in nums:
            s = x - k
            e = x + k
            events.append((s, 0))  # Start event
            events.append((e, 1))  # End event
        
        # Sort the events: first by point, then by type (start before end)
        events.sort()
        
        current = 0
        max_beauty = 0
        for event in events:
            if event[1] == 0:
                current += 1
                if current > max_beauty:
                    max_beauty = current
            else:
                current -= 1
        return max_beauty