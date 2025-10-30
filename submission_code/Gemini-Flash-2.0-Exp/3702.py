from math import gcd
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def lcm(a, b):
            return (a * b) // gcd(a, b)

        max_len = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                sub_array = nums[i:j+1]
                
                product = 1
                for num in sub_array:
                    product *= num
                
                current_gcd = sub_array[0]
                for k in range(1, len(sub_array)):
                    current_gcd = gcd(current_gcd, sub_array[k])
                
                current_lcm = sub_array[0]
                for k in range(1, len(sub_array)):
                    current_lcm = lcm(current_lcm, sub_array[k])
                
                if product == current_lcm * current_gcd:
                    max_len = max(max_len, len(sub_array))
        
        return max_len