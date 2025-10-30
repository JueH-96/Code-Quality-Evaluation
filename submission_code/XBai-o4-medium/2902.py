from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        digits = defaultdict(list)
        for num in nums:
            max_d = max(int(c) for c in str(num))
            digits[max_d].append(num)
        
        max_sum = -1
        for key in digits:
            numbers = digits[key]
            if len(numbers) >= 2:
                sorted_numbers = sorted(numbers, reverse=True)
                current_sum = sorted_numbers[0] + sorted_numbers[1]
                if current_sum > max_sum:
                    max_sum = current_sum
        return max_sum