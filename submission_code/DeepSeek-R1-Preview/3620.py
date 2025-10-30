from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        prev = -float('inf')
        count = 0
        for x in nums:
            target = prev + 1
            low = x - k
            y = max(target, low)
            if y <= x + k:
                prev = y
                count += 1
        return count