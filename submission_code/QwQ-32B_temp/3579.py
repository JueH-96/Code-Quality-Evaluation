from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        for perm in permutations(nums):
            binary_str = ''.join([bin(x)[2:] for x in perm])
            current = int(binary_str, 2)
            if current > max_val:
                max_val = current
        return max_val