from typing import List
from math import gcd
from functools import reduce
from math import lcm

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def compute_gcd(arr):
            return reduce(gcd, arr)

        def compute_lcm(arr):
            return reduce(lcm, arr)

        max_score = 0
        n = len(nums)

        # Check the score without removing any element
        max_score = max(max_score, compute_gcd(nums) * compute_lcm(nums))

        # Check the score by removing each element one by one
        for i in range(n):
            new_nums = nums[:i] + nums[i+1:]  # Create a new array without the i-th element
            if new_nums:  # Ensure the new array is not empty
                current_gcd = compute_gcd(new_nums)
                current_lcm = compute_lcm(new_nums)
                max_score = max(max_score, current_gcd * current_lcm)

        return max_score