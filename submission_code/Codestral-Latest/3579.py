from itertools import permutations
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_value = 0
        for perm in permutations(nums):
            binary_str = ''.join(format(num, 'b') for num in perm)
            decimal_value = int(binary_str, 2)
            max_value = max(max_value, decimal_value)
        return max_value