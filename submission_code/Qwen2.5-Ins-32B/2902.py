from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_digit_sum = {}
        max_sum = -1
        
        for num in sorted(nums, reverse=True):
            max_digit = max(int(digit) for digit in str(num))
            if max_digit in max_digit_sum:
                max_sum = max(max_sum, max_digit_sum[max_digit] + num)
                max_digit_sum[max_digit] = max(max_digit_sum[max_digit], num)
            else:
                max_digit_sum[max_digit] = num
        
        return max_sum