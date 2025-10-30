from typing import List
from itertools import combinations

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        max_strength = float('-inf')
        for comb in combinations(nums, len(nums) - 1):
            strength = 1
            for num in comb:
                strength *= num
            max_strength = max(max_strength, strength)
        
        return max_strength