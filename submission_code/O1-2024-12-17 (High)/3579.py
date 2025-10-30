class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        max_val = 0
        for perm in permutations(nums):
            binary_str = ''.join(bin(x)[2:] for x in perm)
            val = int(binary_str, 2)
            max_val = max(max_val, val)
        
        return max_val