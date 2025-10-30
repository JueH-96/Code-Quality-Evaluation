from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_prod = float('-inf')
        for mask in range(1, 1 << n):
            product = 1
            for j in range(n):
                if mask & (1 << j):
                    product *= nums[j]
            max_prod = max(max_prod, product)
        return max_prod