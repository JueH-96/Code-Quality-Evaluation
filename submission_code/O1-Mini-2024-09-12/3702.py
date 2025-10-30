from math import gcd
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def compute_lcm(a, b):
            return a * b // gcd(a, b)
        
        n = len(nums)
        max_len = 1  # At least one element is always a valid subarray
        
        for i in range(n):
            current_prod = 1
            current_gcd = nums[i]
            current_lcm = nums[i]
            for j in range(i, n):
                current_prod *= nums[j]
                current_gcd = gcd(current_gcd, nums[j])
                current_lcm = compute_lcm(current_lcm, nums[j])
                
                if current_prod == current_gcd * current_lcm:
                    current_length = j - i + 1
                    if current_length > max_len:
                        max_len = current_length
                        
        return max_len