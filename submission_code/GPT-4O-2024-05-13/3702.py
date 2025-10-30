from math import gcd
from functools import reduce
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def prod(arr):
            result = 1
            for num in arr:
                result *= num
            return result
        
        def gcd_arr(arr):
            return reduce(gcd, arr)
        
        def lcm_arr(arr):
            return reduce(lcm, arr)
        
        n = len(nums)
        max_len = 0
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                if prod(subarray) == lcm_arr(subarray) * gcd_arr(subarray):
                    max_len = max(max_len, j - i + 1)
        
        return max_len