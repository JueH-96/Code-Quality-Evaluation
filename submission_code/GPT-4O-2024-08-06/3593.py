from typing import List
from math import gcd
from functools import reduce

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        
        def gcd_of_list(lst):
            return reduce(gcd, lst)
        
        def lcm_of_list(lst):
            return reduce(lcm, lst)
        
        n = len(nums)
        
        if n == 1:
            return nums[0] * nums[0]
        
        # Calculate GCD and LCM of the entire array
        total_gcd = gcd_of_list(nums)
        total_lcm = lcm_of_list(nums)
        
        # Calculate the factor score without removing any element
        max_score = total_gcd * total_lcm
        
        # Try removing each element and calculate the factor score
        for i in range(n):
            # Create a new list without the i-th element
            new_list = nums[:i] + nums[i+1:]
            
            # Calculate GCD and LCM of the new list
            if new_list:
                current_gcd = gcd_of_list(new_list)
                current_lcm = lcm_of_list(new_list)
                current_score = current_gcd * current_lcm
                max_score = max(max_score, current_score)
        
        return max_score