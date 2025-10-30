from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        # Build a sweep-line event counter for intervals [num-k, num+k]
        events = Counter()
        for num in nums:
            left = num - k
            right_plus_one = num + k + 1
            events[left] += 1
            events[right_plus_one] -= 1
        
        # Sweep through the sorted event points, tracking the current overlap
        curr = 0
        ans = 0
        for point in sorted(events):
            curr += events[point]
            if curr > ans:
                ans = curr
        
        return ans