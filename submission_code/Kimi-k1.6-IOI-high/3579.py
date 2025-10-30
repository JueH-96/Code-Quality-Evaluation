from itertools import permutations
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        for p in permutations(nums):
            binary_str = ''.join(bin(n)[2:] for n in p)
            current = int(binary_str, 2)
            if current > max_val:
                max_val = current
        return max_val