from math import gcd
from functools import reduce

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        def factor_score(arr):
            if not arr:
                return 0
            gcd_val = reduce(gcd, arr)
            lcm_val = reduce(lcm, arr)
            return gcd_val * lcm_val

        max_score = factor_score(nums)
        for i in range(len(nums)):
            new_nums = nums[:i] + nums[i+1:]
            max_score = max(max_score, factor_score(new_nums))
        return max_score