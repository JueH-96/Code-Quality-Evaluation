from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        for perm in permutations(nums):
            binary_concat = ''.join(bin(num)[2:] for num in perm)
            num_val = int(binary_concat, 2)
            if num_val > max_num:
                max_num = num_val
        return max_num