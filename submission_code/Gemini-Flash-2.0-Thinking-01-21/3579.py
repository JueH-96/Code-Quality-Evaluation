from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        perms = list(permutations(nums))
        for p in perms:
            binary_string = ""
            for n in p:
                binary_string += bin(n)[2:]
            current_num = int(binary_string, 2)
            max_num = max(max_num, current_num)
        return max_num