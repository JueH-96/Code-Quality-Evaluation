from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        max_value = 0
        for perm in permutations(nums):
            binary_str = ''.join(bin(num)[2:] for num in perm)
            max_value = max(max_value, int(binary_str, 2))
        
        return max_value