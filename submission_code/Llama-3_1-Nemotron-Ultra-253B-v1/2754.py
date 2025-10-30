from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        max_str = -float('inf')
        n = len(nums)
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if mask & (1 << i):
                    product *= nums[i]
            if product > max_str:
                max_str = product
        return max_str