class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        import itertools
        max_num = 0
        for perm in itertools.permutations(nums):
            binary_string = ""
            for num in perm:
                binary_string += bin(num)[2:]
            max_num = max(max_num, int(binary_string, 2))
        return max_num