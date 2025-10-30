from math import gcd
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def lcm(a, b):
            return (a * b) // gcd(a, b)

        def calculate_score(arr):
            if not arr:
                return 0
            
            current_gcd = arr[0]
            current_lcm = arr[0]
            
            for i in range(1, len(arr)):
                current_gcd = gcd(current_gcd, arr[i])
                current_lcm = lcm(current_lcm, arr[i])
            
            return current_gcd * current_lcm

        max_score = 0
        for i in range(len(nums)):
            temp_nums = nums[:i] + nums[i+1:]
            max_score = max(max_score, calculate_score(temp_nums))

        max_score = max(max_score, calculate_score(nums))
        
        return max_score