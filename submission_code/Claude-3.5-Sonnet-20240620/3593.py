from math import gcd
from functools import reduce

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        def array_gcd(arr):
            return reduce(gcd, arr)

        def array_lcm(arr):
            return reduce(lcm, arr)

        n = len(nums)
        if n == 1:
            return nums[0] * nums[0]

        total_gcd = array_gcd(nums)
        total_lcm = array_lcm(nums)
        max_score = total_gcd * total_lcm

        for i in range(n):
            remaining = nums[:i] + nums[i+1:]
            current_gcd = array_gcd(remaining)
            current_lcm = array_lcm(remaining)
            max_score = max(max_score, current_gcd * current_lcm)

        return max_score