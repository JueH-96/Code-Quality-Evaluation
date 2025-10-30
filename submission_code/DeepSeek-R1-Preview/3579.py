from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums):
        max_val = 0
        for perm in permutations(nums):
            bin_str = ''.join(bin(num)[2:] for num in perm)
            current = int(bin_str, 2)
            if current > max_val:
                max_val = current
        return max_val