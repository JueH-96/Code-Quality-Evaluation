from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for a in nums:
            start = a - k
            end = a + k
            events.append((start, 1))
            events.append((end + 1, -1))
        
        events.sort()
        
        max_count = 0
        current_count = 0
        
        for x, delta in events:
            current_count += delta
            if current_count > max_count:
                max_count = current_count
        
        return max_count