from math import gcd
from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(x):
            return int(x**0.5)**2 == x
        
        def can_form_complete_set(a, b):
            return is_perfect_square(a * b)
        
        n = len(nums)
        max_sum = 0
        
        for i in range(n):
            current_sum = nums[i]
            for j in range(i + 1, n):
                if can_form_complete_set(nums[i], nums[j]):
                    current_sum += nums[j]
            max_sum = max(max_sum, current_sum)
        
        return max_sum