from typing import List
from math import gcd
from functools import reduce

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def prod(arr):
            result = 1
            for num in arr:
                result *= num
            return result
        
        n = len(nums)
        max_length = 0
        
        for i in range(n):
            current_gcd = nums[i]
            current_lcm = nums[i]
            current_prod = nums[i]
            
            for j in range(i, n):
                if j > i:
                    current_gcd = gcd(current_gcd, nums[j])
                    current_lcm = lcm(current_lcm, nums[j])
                    current_prod *= nums[j]
                
                if current_prod == current_lcm * current_gcd:
                    max_length = max(max_length, j - i + 1)
        
        return max_length