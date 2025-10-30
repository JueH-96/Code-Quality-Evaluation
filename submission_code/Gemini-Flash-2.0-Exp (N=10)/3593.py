import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        def lcm(a, b):
            if a == 0 or b == 0:
                return 0
            return (a * b) // gcd(a, b)

        def calculate_score(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0] * arr[0]
            
            current_gcd = arr[0]
            current_lcm = arr[0]
            for i in range(1, len(arr)):
                current_gcd = gcd(current_gcd, arr[i])
                current_lcm = lcm(current_lcm, arr[i])
            return current_gcd * current_lcm

        max_score = 0
        for i in range(len(nums) + 1):
            if i == len(nums):
                max_score = max(max_score, calculate_score(nums))
            else:
                temp_nums = nums[:i] + nums[i+1:]
                max_score = max(max_score, calculate_score(temp_nums))
        
        return max_score