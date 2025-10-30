from math import gcd
from functools import reduce
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        n = len(nums)
        max_length = 1
        
        for start in range(n):
            current_gcd = nums[start]
            current_lcm = nums[start]
            for end in range(start, n):
                current_gcd = gcd(current_gcd, nums[end])
                current_lcm = lcm(current_lcm, nums[end])
                
                # Calculate product of the subarray
                if end == start:
                    current_product = nums[start]
                else:
                    current_product *= nums[end]
                
                # Check if the subarray is product equivalent
                if current_product == current_lcm * current_gcd:
                    max_length = max(max_length, end - start + 1)
        
        return max_length