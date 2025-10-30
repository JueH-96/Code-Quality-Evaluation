from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        max_strength = -float('inf')
        n = len(nums)
        for i in range(1, 1 << n):
            current_strength = 1
            group_is_formed = False
            for j in range(n):
                if (i >> j) & 1:
                    current_strength *= nums[j]
                    group_is_formed = True
            if group_is_formed:
                max_strength = max(max_strength, current_strength)
        return max_strength