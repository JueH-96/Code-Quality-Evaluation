from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_value = 0
        
        for i in range(n):
            for j in range(i + k, min(i + 2 * k, n) + 1):
                left_or = 0
                right_or = 0
                for l in range(i, j - k):
                    left_or |= nums[l]
                for r in range(j - k, j):
                    right_or |= nums[r]
                max_value = max(max_value, left_or ^ right_or)
                
        return max_value