from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = -1
        num_map = {}
        
        for num in nums:
            max_digit = max(int(digit) for digit in str(num))
            if max_digit in num_map:
                max_sum = max(max_sum, num + num_map[max_digit])
                num_map[max_digit] = max(num_map[max_digit], num)
            else:
                num_map[max_digit] = num
        
        return max_sum