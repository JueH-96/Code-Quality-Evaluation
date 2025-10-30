import math
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def get_square_free(x):
            factors = {}
            i = 2
            while i * i <= x:
                while x % i == 0:
                    factors[i] = factors.get(i, 0) + 1
                    x = x // i
                i += 1
            if x > 1:
                factors[x] = 1
            square_free = 1
            for p in factors:
                if factors[p] % 2 != 0:
                    square_free *= p
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