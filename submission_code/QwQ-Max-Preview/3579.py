from itertools import permutations
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        for perm in permutations(nums):
            binary_str = ''.join(bin(x)[2:] for x in perm)
            current_num = int(binary_str, 2)
            if current_num > max_num:
                max_num = current_num
        return max_num