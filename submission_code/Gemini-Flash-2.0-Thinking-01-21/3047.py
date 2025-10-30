import math

class Solution:
    def get_square_free_part(self, n):
        square_free_part = 1
        i = 2
        temp_n = n
        while i * i <= temp_n:
            if temp_n % (i * i) == 0:
                while temp_n % (i * i) == 0:
                    temp_n //= (i * i)
            if temp_n % i == 0:
                square_free_part *= i
                temp_n //= i
            i += 1
        if temp_n > 1:
            square_free_part *= temp_n
        return square_free_part

    def maximumSum(self, nums: List[int]) -> int:
        square_free_part_sums = {}
        for num in nums:
            sf_part = self.get_square_free_part(num)
            if sf_part in square_free_part_sums:
                square_free_part_sums[sf_part] += num
            else:
                square_free_part_sums[sf_part] = num
        max_sum = 0
        if not square_free_part_sums:
            return 0
        for sf_part in square_free_part_sums:
            max_sum = max(max_sum, square_free_part_sums[sf_part])
        return max_sum

from typing import List