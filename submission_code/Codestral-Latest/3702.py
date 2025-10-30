from math import gcd
from functools import reduce
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        def prod(arr):
            return reduce(lambda x, y: x * y, arr, 1)

        max_len = 0
        n = len(nums)

        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                sub_prod = prod(subarray)
                sub_gcd = reduce(gcd, subarray)
                sub_lcm = reduce(lcm, subarray)

                if sub_prod == sub_lcm * sub_gcd:
                    max_len = max(max_len, j - i + 1)

        return max_len