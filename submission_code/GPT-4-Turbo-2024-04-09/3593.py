from typing import List
from math import gcd
from functools import reduce
import sys

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        
        def lcm_list(numbers):
            return reduce(lcm, numbers)
        
        def gcd_list(numbers):
            return reduce(gcd, numbers)
        
        if len(nums) == 1:
            return nums[0] * nums[0]
        
        # Calculate the GCD and LCM of the entire array
        full_gcd = gcd_list(nums)
        full_lcm = lcm_list(nums)
        max_score = full_gcd * full_lcm
        
        # Calculate the maximum score after removing each element
        for i in range(len(nums)):
            if len(nums) > 1:
                # Remove element at index i
                sub_nums = nums[:i] + nums[i+1:]
                sub_gcd = gcd_list(sub_nums)
                sub_lcm = lcm_list(sub_nums)
                max_score = max(max_score, sub_gcd * sub_lcm)
        
        return max_score