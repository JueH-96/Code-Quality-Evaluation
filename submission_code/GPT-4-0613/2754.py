from typing import List
from itertools import combinations

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        max_strength = float('-inf')
        for i in range(1, len(nums) + 1):
            for combination in combinations(nums, i):
                strength = 1
                for num in combination:
                    strength *= num
                max_strength = max(max_strength, strength)
        return max_strength