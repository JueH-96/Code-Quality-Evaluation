from math import gcd
from functools import reduce
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def gcd_list(arr):
            return reduce(gcd, arr)
        
        def lcm_list(arr):
            return reduce(lcm, arr)
        
        n = len(nums)
        
        if n == 1:
            return nums[0] * nums[0]
        
        # Calculate the GCD and LCM of the entire array
        total_gcd = gcd_list(nums)
        total_lcm = lcm_list(nums)
        
        # Initial factor score without removing any element
        max_score = total_gcd * total_lcm
        
        # Try removing each element and calculate the factor score
        for i in range(n):
            temp_nums = nums[:i] + nums[i+1:]
            if temp_nums:
                current_gcd = gcd_list(temp_nums)
                current_lcm = lcm_list(temp_nums)
                max_score = max(max_score, current_gcd * current_lcm)
        
        return max_score