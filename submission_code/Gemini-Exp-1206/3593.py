from math import gcd
from typing import List
from functools import reduce

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)

        def array_lcm(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0]
            return reduce(lcm, arr)

        def array_gcd(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0]
            return reduce(gcd, arr)

        max_score = 0
        for i in range(len(nums) + 1):
            temp_nums = nums[:]
            if i < len(nums):
                temp_nums.pop(i)
            
            if not temp_nums:
                current_score = 0
            else:
                current_score = array_lcm(temp_nums) * array_gcd(temp_nums)
            max_score = max(max_score, current_score)

        return max_score