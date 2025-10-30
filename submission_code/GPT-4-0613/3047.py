from typing import List
from collections import defaultdict
from math import isqrt

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def prime_factors(n):
            i = 2
            factors = defaultdict(int)
            while i * i <= n:
                if n % i:
                    i += 1
                else:
                    n //= i
                    factors[i] += 1
            if n > 1:
                factors[n] += 1
            return factors

        def is_square(n):
            return n == isqrt(n) ** 2

        def to_square_free(n):
            factors = prime_factors(n)
            res = 1
            for p, e in factors.items():
                if e % 2 == 1:
                    res *= p
            return res

        square_free_nums = [to_square_free(n) for n in nums]
        counter = defaultdict(int)
        for n in square_free_nums:
            counter[n] += 1

        max_sum = max(nums)
        for n, count in counter.items():
            if count > 1:
                max_sum = max(max_sum, n * count)
        return max_sum