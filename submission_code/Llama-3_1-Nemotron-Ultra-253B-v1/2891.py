from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            a = num - k
            b = num + k + 1  # End event is at b + 1, so here it's num + k + 1
            events.append((a, 1))
            events.append((b, -1))
        
        events.sort()
        
        current = 0
        max_count = 0
        for x, delta in events:
            current += delta
            if current > max_count:
                max_count = current
        
        return max_count