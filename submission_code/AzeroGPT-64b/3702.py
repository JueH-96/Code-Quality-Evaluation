from typing import List
from math import gcd
from functools import reduce

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def product(arr):
            return reduce(lambda x, y: x * y, arr, 1)
        
        max_length = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                subarray = nums[i:j+1]
                if product(subarray) == lcm(reduce(lcm, subarray), 1) * gcd(subarray):
                    max_length = max(max_length, len(subarray))
        return max_length