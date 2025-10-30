from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_val = 0
        for perm in permutations(nums):
            bin_str = ''.join(bin(num)[2:] for num in perm)
            num_val = int(bin_str, 2)
            if num_val > max_val:
                max_val = num_val
        return max_val