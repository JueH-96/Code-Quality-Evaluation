from typing import List
from math import gcd
from functools import reduce

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def compute_gcd(arr):
            return reduce(gcd, arr) if arr else 0
        
        def compute_lcm(arr):
            def lcm(a, b):
                return abs(a * b) // gcd(a, b)
            return reduce(lcm, arr, 1) if arr else 0
        
        n = len(nums)
        max_score = 0
        
        # Consider not removing any element
        current_gcd = compute_gcd(nums)
        current_lcm = compute_lcm(nums)
        max_score = current_gcd * current_lcm
        
        # Consider removing each element one by one
        for i in range(n):
            new_nums = nums[:i] + nums[i+1:]
            if not new_nums:
                score = 0
            else:
                new_gcd = compute_gcd(new_nums)
                new_lcm = compute_lcm(new_nums)
                score = new_gcd * new_lcm
            if score > max_score:
                max_score = score
        
        return max_score