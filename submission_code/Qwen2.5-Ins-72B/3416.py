from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        digit_count = len(str(nums[0]))
        total_diff = 0
        
        for digit_pos in range(digit_count):
            digit_freq = [0] * 10
            for num in nums:
                digit = (num // 10**digit_pos) % 10
                digit_freq[digit] += 1
            
            for i in range(10):
                for j in range(10):
                    if i != j:
                        total_diff += digit_freq[i] * digit_freq[j]
        
        return total_diff