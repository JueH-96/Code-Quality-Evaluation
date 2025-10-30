from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        total_sum = 0
        num_strs = list(map(str, nums))
        num_length = len(num_strs[0])
        
        for i in range(num_length):
            digit_count = [0] * 10
            for num_str in num_strs:
                digit_count[int(num_str[i])] += 1
            
            for j, num_str in enumerate(num_strs):
                digit = int(num_str[i])
                total_sum += j * digit_count[digit] - (len(nums) - j - digit_count[digit]) * digit_count[digit]
        
        return total_sum