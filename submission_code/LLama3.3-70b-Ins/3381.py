from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        min_length = float('inf')
        
        for i in range(len(nums)):
            or_value = 0
            for j in range(i, len(nums)):
                or_value |= nums[j]
                if or_value >= k:
                    min_length = min(min_length, j - i + 1)
        
        return min_length if min_length != float('inf') else -1