from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        for perm in permutations(nums):
            binary_str = ''.join([bin(num)[2:] for num in perm])
            max_num = max(max_num, int(binary_str, 2))
        return max_num