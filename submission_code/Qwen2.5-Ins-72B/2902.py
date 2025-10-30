from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(n):
            return max(int(d) for d in str(n))
        
        max_digit_dict = {}
        max_sum = -1
        
        for num in nums:
            digit = max_digit(num)
            if digit in max_digit_dict:
                max_sum = max(max_sum, max_digit_dict[digit] + num)
                max_digit_dict[digit] = max(max_digit_dict[digit], num)
            else:
                max_digit_dict[digit] = num
        
        return max_sum