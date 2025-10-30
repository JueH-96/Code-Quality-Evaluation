import itertools
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        for perm in itertools.permutations(nums):
            concatenated = ''.join(bin(num)[2:] for num in perm)
            current_value = int(concatenated, 2)
            if current_value > max_val:
                max_val = current_value
        return max_val