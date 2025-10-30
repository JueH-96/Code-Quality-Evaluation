from typing import List
import itertools

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        for perm in itertools.permutations(nums):
            binary_str = ""
            for num in perm:
                binary_str += bin(num)[2:]
            current_val = int(binary_str, 2)
            if current_val > max_val:
                max_val = current_val
        return max_val