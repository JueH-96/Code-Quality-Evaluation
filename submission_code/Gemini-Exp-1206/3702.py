from math import gcd, lcm
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def prod(arr):
            res = 1
            for x in arr:
                res *= x
            return res

        def gcd_arr(arr):
            res = arr[0]
            for x in arr[1:]:
                res = gcd(res, x)
            return res

        def lcm_arr(arr):
            res = arr[0]
            for x in arr[1:]:
                res = lcm(res, x)
            return res

        n = len(nums)
        max_len = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j + 1]
                if prod(sub_array) == lcm_arr(sub_array) * gcd_arr(sub_array):
                    max_len = max(max_len, len(sub_array))
        return max_len