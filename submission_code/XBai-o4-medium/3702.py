import math
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_len = 0
        n = len(nums)
        for i in range(n):
            current_gcd = nums[i]
            current_lcm = nums[i]
            product = nums[i]
            if product == current_lcm * current_gcd:
                max_len = max(max_len, 1)
            for j in range(i + 1, n):
                product *= nums[j]
                current_gcd = math.gcd(current_gcd, nums[j])
                current_lcm = (current_lcm * nums[j]) // math.gcd(current_lcm, nums[j])
                if product == current_lcm * current_gcd:
                    current_length = j - i + 1
                    if current_length > max_len:
                        max_len = current_length
        return max_len