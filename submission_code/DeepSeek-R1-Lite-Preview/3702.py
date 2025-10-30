from math import gcd, lcm
from functools import reduce
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_length = 0
        n = len(nums)
        
        for start in range(n):
            current_gcd = nums[start]
            current_lcm = nums[start]
            current_prod = nums[start]
            
            if current_prod == current_lcm * current_gcd:
                max_length = max(max_length, 1)
            
            for end in range(start + 1, n):
                current_gcd = gcd(current_gcd, nums[end])
                current_lcm = lcm(current_lcm, nums[end])
                current_prod *= nums[end]
                
                if current_prod == current_lcm * current_gcd:
                    max_length = max(max_length, end - start + 1)
        
        return max_length