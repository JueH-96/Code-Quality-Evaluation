class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        max_num = 0
        for p in permutations(nums):
            binary_str = ''
            for num in p:
                binary_str += bin(num)[2:]
            current_num = int(binary_str, 2)
            if current_num > max_num:
                max_num = current_num
        return max_num