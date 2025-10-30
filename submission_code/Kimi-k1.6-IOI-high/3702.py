import math
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_len = 0
        n = len(nums)
        for i in range(n):
            product = nums[i]
            current_gcd = nums[i]
            current_lcm = nums[i]
            # Check for subarray of length 1
            if product == current_gcd * current_lcm:
                if 1 > max_len:
                    max_len = 1
            # Expand subarray to the right
            for j in range(i + 1, n):
                product *= nums[j]
                current_gcd = math.gcd(current_gcd, nums[j])
                # Calculate new LCM
                gcd_lcm_num = math.gcd(current_lcm, nums[j])
                current_lcm = (current_lcm * nums[j]) // gcd_lcm_num
                # Check if the condition is met
                if product == current_gcd * current_lcm:
                    current_length = j - i + 1
                    if current_length > max_len:
                        max_len = current_length
        return max_len