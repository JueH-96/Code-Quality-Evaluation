from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        max_strength = float('-inf')
        n = len(nums)

        for i in range(1, 1 << n):  # Iterate through all non-empty subsets
            current_product = 1
            for j in range(n):
                if (i >> j) & 1:
                    current_product *= nums[j]

            max_strength = max(max_strength, current_product)

        return max_strength