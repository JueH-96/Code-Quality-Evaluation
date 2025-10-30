from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            low = num - k
            high = num + k
            events.append((low, 1))
            events.append((high + 1, -1))
        
        events.sort()
        
        current = 0
        max_beauty = 0
        
        for x, delta in events:
            current += delta
            if current > max_beauty:
                max_beauty = current
        
        return max_beauty