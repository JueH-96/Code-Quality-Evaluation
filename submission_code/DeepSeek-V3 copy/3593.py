from math import gcd
from functools import reduce
from typing import List

class Solution:
    def compute_gcd(self, arr):
        return reduce(gcd, arr)
    
    def compute_lcm(self, arr):
        def lcm(a, b):
            return a * b // gcd(a, b)
        return reduce(lcm, arr, 1)
    
    def maxScore(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_score = 0
        n = len(nums)
        for i in range(n):
            temp = nums[:i] + nums[i+1:]
            if not temp:
                current_score = 0
            else:
                current_gcd = self.compute_gcd(temp)
                current_lcm = self.compute_lcm(temp)
                current_score = current_gcd * current_lcm
            if current_score > max_score:
                max_score = current_score
        # Also consider the case where no element is removed
        current_gcd = self.compute_gcd(nums)
        current_lcm = self.compute_lcm(nums)
        current_score = current_gcd * current_lcm
        if current_score > max_score:
            max_score = current_score
        return max_score