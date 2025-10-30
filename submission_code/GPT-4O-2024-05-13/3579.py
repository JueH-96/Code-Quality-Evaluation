from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        max_num = 0
        for perm in permutations(nums):
            binary_concat = ''.join(bin(num)[2:] for num in perm)
            decimal_value = int(binary_concat, 2)
            max_num = max(max_num, decimal_value)
        
        return max_num