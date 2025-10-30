from math import isqrt
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def get_square_free(x):
            i = 2
            square_free = 1
            while i * i <= x:
                count = 0
                while x % i == 0:
                    x //= i
                    count += 1
                if count % 2 != 0:
                    square_free *= i
                i += 1
            if x != 1:
                square_free *= x
            return square_free
        
        square_free_map = defaultdict(int)
        for idx, num in enumerate(nums, 1):
            sf = get_square_free(num)
            square_free_map[sf] += num
        
        max_sum = 0
        for key in square_free_map:
            if square_free_map[key] > max_sum:
                max_sum = square_free_map[key]
        return max_sum