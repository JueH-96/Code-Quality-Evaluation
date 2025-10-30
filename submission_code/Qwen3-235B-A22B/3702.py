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
            # Check subarray of length 1
            if current_product == current_gcd * current_lcm:
                max_len = max(max_len, 1)
            # Expand the subarray starting at i
            for j in range(i + 1, n):
                current_product *= nums[j]
                current_gcd = math.gcd(current_gcd, nums[j])
                current_lcm = current_lcm * nums[j] // math.gcd(current_lcm, nums[j])
                if current_product == current_gcd * current_lcm:
                    current_length = j - i + 1
                    if current_length > max_len:
                        max_len = current_length
        return max_len