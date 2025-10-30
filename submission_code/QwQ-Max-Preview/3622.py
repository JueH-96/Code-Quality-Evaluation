from bisect import bisect_left, bisect_right
from collections import Counter
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        if numOperations == 0:
            return max(Counter(nums).values()) if nums else 0
        
        nums.sort()
        freq = Counter(nums)
        candidates = set()
        for x in nums:
            candidates.add(x - k)
            candidates.add(x)
            candidates.add(x + k)
        
        max_count = 0
        for T in candidates:
            left_bound = T - k
            right_bound = T + k
            l = bisect_left(nums, left_bound)
            r = bisect_right(nums, right_bound)
            coverage = r - l
            original = freq.get(T, 0)
            current = min(coverage, original + numOperations)
            if current > max_count:
                max_count = current
        return max_count