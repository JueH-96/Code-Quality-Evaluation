from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_number = 0
        for perm in permutations(nums):
            binary_string = ""
            for num in perm:
                binary_string += bin(num)[2:]
            if binary_string:
                max_number = max(max_number, int(binary_string, 2))
        return max_number