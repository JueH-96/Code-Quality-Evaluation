from math import gcd
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_len = 0
        n = len(nums)
        for i in range(n):
            current_product = nums[i]
            current_gcd = nums[i]
            current_lcm = nums[i]
            # Check subarray of length 1
            if current_product == current_lcm * current_gcd:
                current_max = 1
                max_len = max(max_len, current_max)
            else:
                current_max = 0
            # Expand to subarrays starting at i and ending at j
            for j in range(i + 1, n):
                current_product *= nums[j]
                current_gcd = gcd(current_gcd, nums[j])
                current_lcm = self.lcm(current_lcm, nums[j])
                if current_product == current_lcm * current_gcd:
                    current_length = j - i + 1
                    if current_length > current_max:
                        current_max = current_length
                        max_len = max(max_len, current_max)
        return max_len
    
    def lcm(self, a: int, b: int) -> int:
        return a * b // gcd(a, b)