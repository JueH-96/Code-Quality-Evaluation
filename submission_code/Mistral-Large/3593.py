from typing import List
import math
from functools import reduce

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def lcm(a, b):
            return a * b // math.gcd(a, b)

        def array_gcd(nums):
            return reduce(math.gcd, nums)

        def array_lcm(nums):
            return reduce(lcm, nums)

        n = len(nums)
        if n == 1:
            return nums[0] * nums[0]

        total_gcd = array_gcd(nums)
        total_lcm = array_lcm(nums)
        max_score = total_gcd * total_lcm

        for i in range(n):
            removed_element = nums[i]
            remaining_nums = nums[:i] + nums[i+1:]
            remaining_gcd = array_gcd(remaining_nums)
            remaining_lcm = array_lcm(remaining_nums)
            current_score = remaining_gcd * remaining_lcm
            max_score = max(max_score, current_score)

        return max_score