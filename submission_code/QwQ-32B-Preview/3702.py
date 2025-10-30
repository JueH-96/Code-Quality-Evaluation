from typing import List
import math
from functools import reduce

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def gcd_list(lst):
            return reduce(math.gcd, lst)
        
        def lcm(a, b):
            return a * b // math.gcd(a, b)
        
        def lcm_list(lst):
            return reduce(lcm, lst)
        
        max_length = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                prod = math.prod(subarray)
                gcd = gcd_list(subarray)
                lcm = lcm_list(subarray)
                if prod == lcm * gcd:
                    max_length = max(max_length, j - i + 1)
        
        return max_length