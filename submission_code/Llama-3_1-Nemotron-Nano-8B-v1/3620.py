from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        prev = -float('inf')
        count = 0
        for num in nums:
            lower = num - k
            upper = num + k
            candidate = max(prev + 1, lower)
            if candidate > upper:
                break
            prev = candidate
            count += 1
        return count