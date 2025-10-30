from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_length = float('inf')
        
        for i in range(n):
            or_val = 0
            for j in range(i, n):
                or_val |= nums[j]
                if or_val >= k:
                    min_length = min(min_length, j - i + 1)
                    break
        
        return min_length if min_length != float('inf') else -1