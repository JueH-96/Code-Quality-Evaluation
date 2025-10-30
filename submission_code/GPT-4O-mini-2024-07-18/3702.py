from typing import List
from math import gcd
from functools import reduce

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        
        def prod(arr):
            result = 1
            for num in arr:
                result *= num
            return result
        
        n = len(nums)
        max_length = 0
        
        for start in range(n):
            current_gcd = nums[start]
            current_lcm = nums[start]
            current_product = nums[start]
            
            for end in range(start, n):
                if end > start:
                    current_gcd = gcd(current_gcd, nums[end])
                    current_lcm = lcm(current_lcm, nums[end])
                    current_product *= nums[end]
                
                if current_product == current_lcm * current_gcd:
                    max_length = max(max_length, end - start + 1)
        
        return max_length