from math import gcd
from functools import reduce
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def lcm(x, y):
            return x * y // gcd(x, y)

        def prod(arr):
            return reduce(lambda x, y: x * y, arr, 1)

        def gcd_arr(arr):
            return reduce(gcd, arr)

        def lcm_arr(arr):
            return reduce(lcm, arr)

        def is_product_equivalent(arr):
            return prod(arr) == lcm_arr(arr) * gcd_arr(arr)

        n = len(nums)
        max_len = 0

        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                if is_product_equivalent(subarray):
                    max_len = max(max_len, len(subarray))

        return max_len