import math
from functools import reduce
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        # Function to compute GCD of a list
        def gcd_list(lst):
            return reduce(math.gcd, lst)
        
        # Function to compute LCM of a list
        def lcm_list(lst):
            return reduce(lambda x, y: x * y // math.gcd(x, y), lst)
        
        # Compute GCD and LCM of the entire array
        gcd_all = gcd_list(nums)
        lcm_all = lcm_list(nums)
        max_score = gcd_all * lcm_all
        
        # Try removing each element and compute the score
        for i in range(len(nums)):
            new_nums = nums[:i] + nums[i+1:]
            if len(new_nums) == 0:
                score = 0
            else:
                gcd_new = gcd_list(new_nums)
                lcm_new = lcm_list(new_nums)
                score = gcd_new * lcm_new
            if score > max_score:
                max_score = score
        
        return max_score