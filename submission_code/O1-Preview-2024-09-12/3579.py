class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        max_num = 0
        for perm in permutations(nums):
            bin_strings = [bin(n)[2:] for n in perm]
            concat_bin = ''.join(bin_strings)
            num = int(concat_bin, 2)
            max_num = max(max_num, num)
        return max_num