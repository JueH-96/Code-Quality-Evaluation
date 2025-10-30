from math import gcd
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_len = 0
        n = len(nums)
        for i in range(n):
            product = nums[i]
            current_gcd = nums[i]
            current_lcm = nums[i]
            if product == current_gcd * current_lcm:
                max_len = max(max_len, 1)
            for j in range(i + 1, n):
                product *= nums[j]
                current_gcd = gcd(current_gcd, nums[j])
                # Calculate LCM of current_lcm and nums[j]
                lcm_value = current_lcm * nums[j] // gcd(current_lcm, nums[j])
                current_lcm = lcm_value
                if product == current_gcd * current_lcm:
                    current_length = j - i + 1
                    if current_length > max_len:
                        max_len = current_length
        return max_len