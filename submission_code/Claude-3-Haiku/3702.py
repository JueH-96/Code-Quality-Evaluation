from typing import List
from math import gcd, lcm

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = 0

        for i in range(n):
            product = 1
            gcd_val = nums[i]
            lcm_val = nums[i]

            for j in range(i, n):
                product *= nums[j]
                gcd_val = gcd(gcd_val, nums[j])
                lcm_val = lcm(lcm_val, nums[j])

                if product == gcd_val * lcm_val:
                    max_length = max(max_length, j - i + 1)

        return max_length