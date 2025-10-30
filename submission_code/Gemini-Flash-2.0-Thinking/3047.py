from typing import List
from math import sqrt

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)

        def get_square_free(num):
            if num == 0:
                return 0
            d = 2
            res = 1
            temp = num
            while d * d <= temp:
                if temp % d == 0:
                    count = 0
                    while temp % d == 0:
                        count += 1
                        temp //= d
                    if count % 2 == 1:
                        res *= d
                d += 1
            if temp > 1:
                res *= temp
            return res

        groups = {}
        for i in range(n):
            sf = get_square_free(nums[i])
            if sf not in groups:
                groups[sf] = 0
            groups[sf] += nums[i]

        max_sum = 0
        for sf in groups:
            max_sum = max(max_sum, groups[sf])

        return max_sum