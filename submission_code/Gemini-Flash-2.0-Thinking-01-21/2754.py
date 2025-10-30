from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_strength = -float('inf')
        for i in range(1, 1 << n):
            current_product = 1
            is_empty_group = True
            for j in range(n):
                if (i >> j) & 1:
                    current_product *= nums[j]
                    is_empty_group = False
            if not is_empty_group:
                max_strength = max(max_strength, current_product)
        if max_strength == -float('inf'):
            return 0
        return max_strength