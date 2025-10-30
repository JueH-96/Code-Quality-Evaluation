from math import gcd
from functools import reduce
from typing import List

def lcm(a, b):
    return a * b // gcd(a, b)

def compute_gcd(arr):
    return reduce(gcd, arr)

def compute_lcm(arr):
    return reduce(lcm, arr)

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        max_score = 0
        for i in range(n):
            temp = nums[:i] + nums[i+1:]
            if not temp:
                current_score = 0
            else:
                current_gcd = compute_gcd(temp)
                current_lcm = compute_lcm(temp)
                current_score = current_gcd * current_lcm
            if current_score > max_score:
                max_score = current_score
        # Also consider the case where no element is removed
        current_gcd = compute_gcd(nums)
        current_lcm = compute_lcm(nums)
        current_score = current_gcd * current_lcm
        if current_score > max_score:
            max_score = current_score
        return max_score