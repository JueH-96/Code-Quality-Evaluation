from typing import List
import itertools

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        permutations = list(itertools.permutations(nums))
        for perm in permutations:
            binary_string = ""
            for num in perm:
                binary_string += bin(num)[2:]
            current_num = int(binary_string, 2)
            max_num = max(max_num, current_num)
        return max_num