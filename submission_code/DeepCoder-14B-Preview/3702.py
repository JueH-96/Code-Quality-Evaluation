import math
from typing import List

class Solution:
    def compute_gcd(self, arr):
        current_gcd = arr[0]
        for num in arr[1:]:
            current_gcd = math.gcd(current_gcd, num)
            if current_gcd == 1:
                break
        return current_gcd

    def compute_lcm(self, arr):
        if not arr:
            return 0
        current_lcm = arr[0]
        for num in arr[1:]:
            current_gcd = math.gcd(current_lcm, num)
            current_lcm = (current_lcm * num) // current_gcd
        return current_lcm

    def maxLength(self, nums: List[int]) -> int:
        max_len = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub = nums[i:j+1]
                product = 1
                for num in sub:
                    product *= num
                gcd_sub = self.compute_gcd(sub)
                lcm_sub = self.compute_lcm(sub)
                if product == gcd_sub * lcm_sub:
                    current_len = j - i + 1
                    if current_len > max_len:
                        max_len = current_len
        return max_len