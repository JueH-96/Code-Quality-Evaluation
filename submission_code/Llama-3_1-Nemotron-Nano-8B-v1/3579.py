from itertools import permutations
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_bin = ''
        for perm in permutations(nums):
            current = ''.join(bin(num)[2:] for num in perm)
            if current > max_bin:
                max_bin = current
        return int(max_bin, 2)