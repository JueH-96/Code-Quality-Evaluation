from math import gcd
from functools import reduce

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        def is_product_equivalent(arr):
            prod = reduce(lambda x, y: x * y, arr)
            arr_gcd = reduce(gcd, arr)
            arr_lcm = reduce(lcm, arr)
            return prod == arr_lcm * arr_gcd

        n = len(nums)
        max_length = 0

        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                if is_product_equivalent(subarray):
                    max_length = max(max_length, j - i + 1)

        return max_length