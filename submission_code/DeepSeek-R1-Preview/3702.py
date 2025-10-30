import math
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = 0

        for i in range(n):
            current_product = nums[i]
            current_gcd = nums[i]
            current_lcm = nums[i]
            if current_product == current_gcd * current_lcm:
                max_len = max(max_len, 1)
            for j in range(i + 1, n):
                current_product *= nums[j]
                current_gcd = math.gcd(current_gcd, nums[j])
                current_lcm = self.lcm(current_lcm, nums[j])
                if current_product == current_gcd * current_lcm:
                    length = j - i + 1
                    if length > max_len:
                        max_len = length
        return max_len

    def lcm(self, a: int, b: int) -> int:
        return a * b // math.gcd(a, b)