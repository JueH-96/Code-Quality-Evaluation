import itertools

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        for perm in itertools.permutations(nums):
            binary_str = ''.join([bin(num)[2:] for num in perm])
            current = int(binary_str, 2)
            if current > max_num:
                max_num = current
        return max_num