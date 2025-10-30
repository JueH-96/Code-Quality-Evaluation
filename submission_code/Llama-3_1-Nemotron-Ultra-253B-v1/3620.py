from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        prev = -float('inf')
        count = 0
        for x in nums:
            low = x - k
            high = x + k
            candidate = max(low, prev + 1)
            if candidate <= high:
                count += 1
                prev = candidate
        return count