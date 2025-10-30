from typing import List
import math

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = 0

        def lcm(a: int, b: int) -> int:
            return a * b // math.gcd(a, b)

        for i in range(n):
            prod = 1
            current_gcd = 0
            current_lcm = 1
            for j in range(i, n):
                num = nums[j]
                prod *= num
                if j == i:
                    current_gcd = num
                    current_lcm = num
                else:
                    current_gcd = math.gcd(current_gcd, num)
                    current_lcm = lcm(current_lcm, num)
                # Check if current subarray is product equivalent
                if prod == current_gcd * current_lcm:
                    max_len = max(max_len, j - i + 1)

        return max_len