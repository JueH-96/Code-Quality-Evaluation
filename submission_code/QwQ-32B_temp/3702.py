import math
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_len = 0
        n = len(nums)
        for left in range(n):
            current_product = 1
            current_gcd = 0
            current_lcm = 1
            for right in range(left, n):
                num = nums[right]
                current_product *= num
                current_gcd = math.gcd(current_gcd, num)
                current_lcm = (current_lcm * num) // math.gcd(current_lcm, num)
                if current_product == current_gcd * current_lcm:
                    current_length = right - left + 1
                    if current_length > max_len:
                        max_len = current_length
        return max_len