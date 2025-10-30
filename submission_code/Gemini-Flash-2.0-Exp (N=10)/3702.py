import math
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        def lcm(a, b):
            if a == 0 or b == 0:
                return 0
            return (a * b) // gcd(a, b)

        def calculate_prod(arr):
            prod = 1
            for num in arr:
                prod *= num
            return prod

        def calculate_gcd(arr):
            if not arr:
                return 0
            result = arr[0]
            for i in range(1, len(arr)):
                result = gcd(result, arr[i])
            return result

        def calculate_lcm(arr):
            if not arr:
                return 0
            result = arr[0]
            for i in range(1, len(arr)):
                result = lcm(result, arr[i])
            return result

        max_len = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                sub_array = nums[i:j+1]
                prod_val = calculate_prod(sub_array)
                gcd_val = calculate_gcd(sub_array)
                lcm_val = calculate_lcm(sub_array)
                if lcm_val == 0:
                    continue
                if prod_val == lcm_val * gcd_val:
                    max_len = max(max_len, len(sub_array))
        return max_len