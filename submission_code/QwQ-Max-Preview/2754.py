from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_strength = float('-inf')
        for mask in range(1, 1 << n):
            product = 1
            for i in range(n):
                if mask & (1 << i):
                    product *= nums[i]
            max_strength = max(max_strength, product)
        return max_strength